# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, rules of 16:40 / 17:40: binary_ng and the opted-in kernels at shapes Blackhole models run.

- Llama 3.1-8B on P150 (tt_transformers): one decoder layer with dummy weights (ModelArgs(dummy_weights=True), the checked-in
  config), decode at batch 1 and 32 and prefill at 128 and 2048, performance and accuracy precision, and the P150 sampling path
  (TTSampling). The residual adds, the MLP multiply with SiLU, the RMSNorms, rotary, SDPA decode and prefill run as the model
  runs them.
- SDXL base on Blackhole (1024x1024 BH model config): resnet blocks and transformer models with random weights (diffusers key
  layout), and the UNet's time-embedding add with SiLU.
- Op replicas of the binary_ng and softmax configurations the r10 inventory read from Gemma4 E2B, Whisper, gpt-oss and
  MiniMax-M3 (unit-test shapes there are regression checks only).
"""
import gc
import os

import pytest
import torch
import ttnn

os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def _rand(*shape):
    return torch.randn(*shape) * 0.02


@pytest.fixture
def mcache(mesh_device):
    c = {}
    yield c
    c.clear()
    gc.collect()


@pytest.fixture
def dcache(device):
    c = {}
    yield c
    c.clear()
    gc.collect()


# ---------------------------------------------------------------- Llama 3.1-8B (tt_transformers, P150)


def _llama_args(mesh_device, opt, batch, max_seq_len):
    from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

    prec = DecodersPrecision.performance if opt == "performance" else DecodersPrecision.accuracy
    args = ModelArgs(
        mesh_device,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        optimizations=lambda ma: prec(ma.n_layers, ma.model_name),
        dummy_weights=True,
        use_hf_rope=False,
    )
    args.n_layers = 1
    return args


def _llama_block(mesh_device, args, transformation_mats, batch):
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.decoder import TransformerBlock

    torch.manual_seed(0)
    state_dict = args.load_state_dict()
    paged = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    perm = torch.randperm(paged.max_num_blocks)
    page_table = torch.argsort(perm).reshape(batch, paged.max_num_blocks // batch)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    block = TransformerBlock(
        args=args,
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=None,
        transformation_mats=transformation_mats,
        paged_attention_config=paged,
    )
    del state_dict
    gc.collect()
    return block, page_table_tt


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("opt, batch", [("performance", 1), ("performance", 32), ("accuracy", 32)])
def test_llama8b_decode(mesh_device, mcache, opt, batch):
    from models.tt_transformers.tt.common import Mode
    from models.tt_transformers.tt.rope import RotarySetup

    c = mcache
    if not c:
        args = _llama_args(mesh_device, opt, batch, 256)
        rope = RotarySetup(
            mesh_device,
            args.max_batch_size,
            args.head_dim,
            args.max_seq_len,
            args.rope_theta,
            args.rope_scaling,
            args.use_qk_fused,
        )
        block, page_table_tt = _llama_block(mesh_device, args, rope.get_both_trans_mats(), batch)
        c.update(args=args, rope=rope, block=block, page_table=page_table_tt)
    args, rope, block = c["args"], c["rope"], c["block"]
    torch.manual_seed(1)
    for step in range(3):
        x = torch.rand(batch, 1, args.dim) * 2 - 1
        x_tt = args.prepare_residual_tensor_decode(x, args.get_residual_mem_config(Mode.DECODE, None))
        pos = torch.tensor([step] * batch)
        pos_tt = ttnn.from_torch(pos, device=mesh_device, dtype=ttnn.int32)
        out = block(
            x_tt,
            pos_tt,
            rot_mats_global=rope.get_rot_mats(pos),
            rot_mats_local=None,
            mode=Mode.DECODE,
            page_table=c["page_table"],
        )
        ttnn.deallocate(out)
        ttnn.deallocate(pos_tt)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("opt, seq", [("performance", 128), ("performance", 2048), ("accuracy", 2048)])
def test_llama8b_prefill(mesh_device, mcache, opt, seq):
    from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
    from models.tt_transformers.tt.rope import get_rot_mats

    c = mcache
    if not c:
        args = _llama_args(mesh_device, opt, 1, seq)
        trans = ttnn.as_tensor(
            get_rot_transformation_mat(args.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rot = get_rot_mats(
            head_dim=args.head_dim, device=mesh_device, seq_len=seq, theta=args.rope_theta, rope_scaling=args.rope_scaling
        )
        block, page_table_tt = _llama_block(mesh_device, args, {"prefill": trans}, 1)
        c.update(args=args, rot=rot, block=block, page_table=page_table_tt, trans=trans)
    args, block = c["args"], c["block"]
    torch.manual_seed(1)
    x = torch.rand(1, seq, args.dim) * 2 - 1
    x_tt = args.prepare_residual_tensor_prefill(x)
    out = block(
        x_tt,
        None,
        rot_mats_global=c["rot"],
        rot_mats_local=None,
        user_id=0,
        mode=Mode.PREFILL,
        page_table=c["page_table"],
    )
    ttnn.deallocate(out)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_llama8b_sampling(mesh_device, mcache):
    from models.common.sampling.tt_sampling import TTSampling
    from models.tt_transformers.tt.ccl import TT_CCL

    c = mcache
    if not c:
        args = _llama_args(mesh_device, "performance", 32, 256)
        sampler = TTSampling(
            mesh_device=mesh_device,
            tt_ccl=TT_CCL(mesh_device),
            args=args,
            k=torch.full((32,), 32, dtype=torch.int64),
            p=torch.full((32,), 0.9, dtype=torch.float32),
            temp=torch.full((32,), 1 / 0.6, dtype=torch.float32),
        )
        vocab = getattr(args, "padded_vocab_size", None) or args.vocab_size
        torch.manual_seed(2)
        logits = ttnn.from_torch(
            torch.randn(1, 1, 32, vocab),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        c.update(args=args, sampler=sampler, logits=logits)
    for _ in range(3):
        res = c["sampler"](c["logits"])
        del res


# ---------------------------------------------------------------- SDXL base, Blackhole 1024x1024 model config


def _sdxl_resnet_sd(path, cin, cout):
    sd = {
        f"{path}.norm1.weight": 1 + _rand(cin),
        f"{path}.norm1.bias": _rand(cin),
        f"{path}.conv1.weight": _rand(cout, cin, 3, 3),
        f"{path}.conv1.bias": _rand(cout),
        f"{path}.time_emb_proj.weight": _rand(cout, 1280),
        f"{path}.time_emb_proj.bias": _rand(cout),
        f"{path}.norm2.weight": 1 + _rand(cout),
        f"{path}.norm2.bias": _rand(cout),
        f"{path}.conv2.weight": _rand(cout, cout, 3, 3),
        f"{path}.conv2.bias": _rand(cout),
    }
    if cin != cout:
        sd[f"{path}.conv_shortcut.weight"] = _rand(cout, cin, 1, 1)
        sd[f"{path}.conv_shortcut.bias"] = _rand(cout)
    return sd


def _sdxl_transformer_sd(path, dim, layers, cross=2048):
    sd = {
        f"{path}.norm.weight": 1 + _rand(dim),
        f"{path}.norm.bias": _rand(dim),
        f"{path}.proj_in.weight": _rand(dim, dim),
        f"{path}.proj_in.bias": _rand(dim),
        f"{path}.proj_out.weight": _rand(dim, dim),
        f"{path}.proj_out.bias": _rand(dim),
    }
    for i in range(layers):
        b = f"{path}.transformer_blocks.{i}"
        for n in ("norm1", "norm2", "norm3"):
            sd[f"{b}.{n}.weight"] = 1 + _rand(dim)
            sd[f"{b}.{n}.bias"] = _rand(dim)
        for a, kv in (("attn1", dim), ("attn2", cross)):
            sd[f"{b}.{a}.to_q.weight"] = _rand(dim, dim)
            sd[f"{b}.{a}.to_k.weight"] = _rand(dim, kv)
            sd[f"{b}.{a}.to_v.weight"] = _rand(dim, kv)
            sd[f"{b}.{a}.to_out.0.weight"] = _rand(dim, dim)
            sd[f"{b}.{a}.to_out.0.bias"] = _rand(dim)
        sd[f"{b}.ff.net.0.proj.weight"] = _rand(8 * dim, dim)
        sd[f"{b}.ff.net.0.proj.bias"] = _rand(8 * dim)
        sd[f"{b}.ff.net.2.weight"] = _rand(dim, 4 * dim)
        sd[f"{b}.ff.net.2.bias"] = _rand(dim)
    return sd


def _sdxl_l1_small():
    from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE_BH

    return SDXL_L1_SMALL_SIZE_BH


SDXL_RESNETS = [
    ("down_blocks", 0, 0, 320, 320, 128),
    ("down_blocks", 1, 1, 640, 640, 64),
    ("down_blocks", 2, 1, 1280, 1280, 32),
    ("up_blocks", 1, 1, 1280, 640, 64),
    ("up_blocks", 2, 1, 640, 320, 128),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 38000}], indirect=True)
@pytest.mark.parametrize("cfg", SDXL_RESNETS, ids=[f"{b[0][0]}{b[1]}r{b[2]}" for b in SDXL_RESNETS])
def test_sdxl_resnet(device, dcache, cfg):
    from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
    from models.demos.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D

    assert _sdxl_l1_small() == 38000
    block, bid, rid, cin, cout, hw = cfg
    c = dcache
    if not c:
        torch.manual_seed(0)
        path = f"{block}.{bid}.resnets.{rid}"
        c["mod"] = TtResnetBlock2D(
            device, _sdxl_resnet_sd(path, cin, cout), path, load_model_optimisations((1024, 1024)), cin != cout
        )
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.rand(1, cin, hw, hw) * 0.2 - 0.1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.reshape(ttnn.permute(x, (0, 2, 3, 1)), (1, 1, hw * hw, cin))
    temb = ttnn.silu(
        ttnn.from_torch(
            torch.rand(1, 1280) * 0.2 - 0.1,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    )
    out, _ = c["mod"].forward(x, temb, [1, cin, hw, hw])
    ttnn.deallocate(out)
    ttnn.deallocate(temb)


SDXL_TRANSFORMERS = [(1, 640, 10, 2, 64), (2, 1280, 20, 10, 32)]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 38000}], indirect=True)
@pytest.mark.parametrize("cfg", SDXL_TRANSFORMERS, ids=["d1", "d2"])
def test_sdxl_transformer(device, dcache, cfg):
    from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
    from models.demos.stable_diffusion_xl_base.tt.tt_transformermodel import TtTransformer2DModel

    bid, dim, heads, layers, hw = cfg
    c = dcache
    if not c:
        torch.manual_seed(0)
        path = f"down_blocks.{bid}.attentions.0"
        c["mod"] = TtTransformer2DModel(
            device, _sdxl_transformer_sd(path, dim, layers), path, load_model_optimisations((1024, 1024)), dim, heads, dim
        )
        c["enc"] = ttnn.from_torch(
            torch.rand(1, 77, 2048) * 0.2 - 0.1,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.rand(1, dim, hw, hw) * 0.2 - 0.1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.reshape(ttnn.permute(x, (0, 2, 3, 1)), (1, 1, hw * hw, dim))
    out = c["mod"].forward(x, [1, dim, hw, hw], None, c["enc"])
    ttnn.deallocate(out)


SDXL_REFINER_GEGLU = [
    ("down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 4096, 768),
    ("down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 1024, 1536),
    ("mid_block.attentions.0.transformer_blocks.0.ff.net.0", 256, 1536),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 38000}], indirect=True)
@pytest.mark.parametrize("cfg", SDXL_REFINER_GEGLU, ids=["d1", "d2", "mid"])
def test_sdxl_refiner_geglu(device, dcache, cfg):
    # The refiner's down_blocks.1 GEGLU multiplies an L1 block-sharded gate by an L1-interleaved input (tt_geglu.py)
    from models.demos.stable_diffusion_xl_base.refiner.tt.model_configs import load_refiner_model_optimisations
    from models.demos.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU

    path, n, dim = cfg
    c = dcache
    if not c:
        torch.manual_seed(0)
        sd = {f"{path}.proj.weight": _rand(8 * dim, dim), f"{path}.proj.bias": _rand(8 * dim)}
        c["mod"] = TtGEGLU(device, sd, path, load_refiner_model_optimisations((1024, 1024)))
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.rand(1, 1, n, dim) * 0.2 - 0.1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(c["mod"].forward(x))


def test_mul_tg(device):
    # llama3_70b_galaxy decode (Llama 3.3-70B, Qwen3-32B on a Blackhole Galaxy), one chip's ff1ff3: bfp8 width shards of
    # [32, 32] on 30 cores of the model's sub-core grids from (1, 0), SiLU on a, bfp8 out (llama_mlp.py)
    grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 30, grids, row_wise=True)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(1)
    mk = lambda: ttnn.from_torch(
        torch.rand(1, 1, 32, 960) * 2 - 1, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc
    )
    a, b = mk(), mk()
    for _ in range(4):
        out = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=mc)
        ttnn.deallocate(out)


@pytest.mark.parametrize("batch", [1, 2])
def test_sdxl_temb_add(device, batch):
    # tt_unet.py: temb = ttnn.add_(temb, temb_add, activations=[SILU]) on the two embedding linears' outputs
    torch.manual_seed(1)
    mk = lambda: ttnn.from_torch(
        torch.rand(batch, 1280) - 0.5,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    a, b = mk(), mk()
    for _ in range(4):
        a = ttnn.add_(a, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])
    ttnn.deallocate(a)
    ttnn.deallocate(b)


# ---------------------------------------------------------------- op replicas (r10 inventory)

_SWI_A = [
    ttnn.UnaryWithParam(ttnn.UnaryOpType.CLAMP_TSS, -7.0, 7.0),
    ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1.0),
]
_SWI_B = [
    ttnn.UnaryWithParam(ttnn.UnaryOpType.CLAMP_TSS, -1.0e30, 7.0),
    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.702),
    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / 1.702),
]

# (id, a shape, b shape or a python scalar, a dtype, b dtype, out dtype, lhs activations, rhs activations)
MUL_CFGS = [
    ("gemma4_mlp_s1", [1, 1, 1, 6144], [1, 1, 1, 6144], "bf16", "bf16", None, None, None),
    ("gemma4_mlp_s32", [1, 1, 32, 6144], [1, 1, 32, 6144], "bf16", "bf16", None, None, None),
    ("gemma4_mlp_s128", [1, 1, 128, 6144], [1, 1, 128, 6144], "bf16", "bf16", None, None, None),
    ("gemma4_mlp_s1024", [1, 1, 1024, 6144], [1, 1, 1024, 6144], "bf16", "bf16", None, None, None),
    ("gemma4_mlp_s4096", [1, 1, 4096, 6144], [1, 1, 4096, 6144], "bf16", "bf16", None, None, None),
    ("gemma4_scale_1536", [1, 32, 1536], 0.125, "bf16", None, None, None, None),
    ("gemma4_scale_256", [1, 1, 32, 256], 0.0625, "bf16", None, None, None, None),
    ("whisper_q_b1_1500", [1, 20, 1500, 64], 0.125, "bf16", None, None, None, None),
    ("whisper_q_b2_1500", [2, 20, 1500, 64], 0.125, "bf16", None, None, None, None),
    ("whisper_q_b1_32", [1, 20, 32, 64], 0.125, "bf16", None, None, None, None),
    ("whisper_q_b2_32", [2, 20, 32, 64], 0.125, "bf16", None, None, None, None),
    ("swiglu_128x12288", [1, 1, 128, 12288], [1, 1, 128, 12288], "bf16", "bf16", None, _SWI_A, _SWI_B),
    ("swiglu_128x3072", [1, 1, 128, 3072], [1, 1, 128, 3072], "bf16", "bf16", None, _SWI_A, _SWI_B),
    ("swiglu_32x3072", [1, 1, 32, 3072], [1, 1, 32, 3072], "bf16", "bf16", None, _SWI_A, _SWI_B),
    ("swiglu_640x768", [1, 1, 640, 768], [1, 1, 640, 768], "bf16", "bf16", None, _SWI_A, _SWI_B),
    ("mul_640x768", [1, 1, 640, 768], [1, 1, 640, 768], "bf16", "bf16", None, None, None),
    ("mul_128x12288", [1, 1, 128, 12288], [1, 1, 128, 12288], "bf16", "bf16", None, None, None),
    ("mul_silu_128x3072_bfp8", [1, 1, 128, 3072], [1, 1, 128, 3072], "bf16", "bf16", "bfp8", "silu", None),
    ("mul_bfp8_32x103424", [32, 103424], [32, 103424], "bfp8", "bf16", None, None, None),
    ("scale_128x3072", [1, 1, 128, 3072], 1.702, "bf16", None, None, None, None),
    ("scale_640x768", [1, 1, 640, 768], 1.702, "bf16", None, None, None, None),
]
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, None: None}


def _dev(t, dt, device):
    return ttnn.from_torch(t, dtype=_DT[dt], device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("cfg", MUL_CFGS, ids=[c[0] for c in MUL_CFGS])
def test_mul_cfg(device, cfg):
    _, sa, sb, da, db, dc, lhs, rhs = cfg
    torch.manual_seed(1)
    a = _dev(torch.rand(sa) * 2 - 1, da, device)
    b = sb if isinstance(sb, float) else _dev(torch.rand(sb) * 2 - 1, db, device)
    kw = {}
    if dc is not None:
        kw["dtype"] = _DT[dc]
    if lhs == "silu":
        kw["input_tensor_a_activations"] = [ttnn.UnaryOpType.SILU]
    elif lhs is not None:
        kw["input_tensor_a_activations"] = lhs
    if rhs is not None:
        kw["input_tensor_b_activations"] = rhs
    for _ in range(4):
        ttnn.deallocate(ttnn.multiply(a, b, **kw))


# (id, shape, numeric_stable)
SOFTMAX_CFGS = [
    ("whisper_b1", [1, 20, 32, 32], False),
    ("whisper_b2", [2, 20, 32, 32], False),
    ("gptoss_route_640", [1, 1, 640, 4], True),
    ("gate_640x256", [1, 1, 640, 256], True),
]


@pytest.mark.parametrize("cfg", SOFTMAX_CFGS, ids=[c[0] for c in SOFTMAX_CFGS])
def test_softmax_cfg(device, cfg):
    _, shape, stable = cfg
    torch.manual_seed(1)
    x = _dev(torch.randn(shape), "bf16", device)
    for _ in range(4):
        ttnn.deallocate(ttnn.softmax(x, dim=-1, numeric_stable=stable, memory_config=ttnn.DRAM_MEMORY_CONFIG))


# ---------------------------------------------------------------- QuietBox 2 decode residual adds, one chip (sixth review)


def _qb2_mc(device, cfg):
    if cfg.startswith("gemma"):
        # gemma4_31b_qb2/tt/decoder.py _mem(5376, 28): width shards of 32x192 on a 7x4 grid
        return ttnn.create_sharded_memory_config(
            (32, 192),
            ttnn.CoreGrid(x=7, y=4),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    if cfg == "llama_qb2":
        # llama31_8b_qb2/tt/decoder.py _width_memcfg(1024, 8): 32x128 on the first 8 cores, row-wise
        grid = device.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(8, grid, True)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
    # qwen38_27b_qb2/tt/decoder.py _residual_memory, residual_cores 80: 32x64 on a 10x8 grid
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, [32, 64], ttnn.ShardOrientation.ROW_MAJOR)
    )


QB2_WIDTH = {"gemma_post": 5376, "gemma_add": 5376, "llama_qb2": 1024, "qwen_qb2": 5120}


@pytest.mark.parametrize("cfg", list(QB2_WIDTH))
def test_qb2_add(device, cfg):
    # gemma_post: decoder.py:676-681 (MUL_UNARY_SFPU by the layer scalar after the add); gemma_add: decoder.py:668;
    # llama_qb2: llama31_8b_qb2 decoder.py:578-582, 597-601; qwen_qb2: qwen38_27b_qb2 decoder.py:548-550, 588-590
    mc = _qb2_mc(device, cfg)
    torch.manual_seed(1)
    mk = lambda: ttnn.from_torch(
        torch.rand(1, 1, 32, QB2_WIDTH[cfg]) - 0.5, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc
    )
    a, b = mk(), mk()
    kw = dict(memory_config=mc, dtype=ttnn.bfloat16)
    if cfg == "gemma_post":
        kw["activations"] = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 0.6875)]
    for _ in range(8):
        ttnn.deallocate(ttnn.add(a, b, **kw))
    ttnn.deallocate(a)
    ttnn.deallocate(b)
