# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Headroom round, eltwise binary (#58818): binary_ng's block sections at the shapes Blackhole models run them.

- SDXL base TtTransformer2DModel down 1 and 2 (GEGLU multiply 4096x2560 and 1024x5120, transformer adds), TtResnetBlock2D
  (resnet adds), the refiner's TtGEGLU (down 2 and mid; down 1 multiplies by an L1-interleaved input: not a block section),
  random weights in diffusers' key layout, the 1024x1024 Blackhole model configs (copied from round 3's test_eb_r11.py).
- Bits: binary_ng add, subtract and multiply on block shards at those shapes, random and special bf16 values; every output
  is hashed by eb_bits_plugin.
Not for merge."""
import gc
import os
import os as _os_guard, sys as _sys_guard

if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")
import pytest
import torch
import ttnn


def _rand(*shape):
    return torch.randn(*shape) * 0.02


@pytest.fixture
def dcache(device):
    c = {}
    yield c
    c.clear()
    gc.collect()


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


SDXL_RESNETS = [
    ("down_blocks", 0, 0, 320, 320, 128),
    ("down_blocks", 2, 1, 1280, 1280, 32),
    ("up_blocks", 2, 1, 640, 320, 128),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 38000}], indirect=True)
@pytest.mark.parametrize("cfg", SDXL_RESNETS, ids=[f"{b[0][0]}{b[1]}r{b[2]}" for b in SDXL_RESNETS])
def test_sdxl_resnet(device, dcache, cfg):
    from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
    from models.demos.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D

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
    ttnn.to_torch(out)
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
    ttnn.to_torch(out)
    ttnn.deallocate(out)


SDXL_REFINER_GEGLU = [
    ("down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0", 4096, 768),
    ("down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0", 1024, 1536),
    ("mid_block.attentions.0.transformer_blocks.0.ff.net.0", 256, 1536),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 38000}], indirect=True)
@pytest.mark.parametrize("cfg", SDXL_REFINER_GEGLU, ids=["d1", "d2", "mid"])
def test_sdxl_refiner_geglu(device, dcache, cfg):
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
    out = c["mod"].forward(x)
    ttnn.to_torch(out)
    ttnn.deallocate(out)


# ---------------------------------------------------------------- bits: binary_ng block sections, special values

_SPECIAL = torch.tensor(
    [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 3.3895e38, -3.3895e38, 1.1755e-38, 9.1835e-41, -9.1835e-41],
    dtype=torch.float32,
)
BITS = {
    # name: shape, grid (y, x), ops
    "geglu_4096x2560": ((1, 1, 4096, 2560), (8, 10), ("mul",)),
    "geglu_1024x5120": ((1, 1, 1024, 5120), (8, 10), ("mul",)),
    "resadd_16384x320": ((1, 1, 16384, 320), (8, 10), ("add", "sub")),
    "tadd_1024x1280": ((1, 1, 1024, 1280), (8, 10), ("add", "sub", "mul")),
}


def _operand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g) * 4
    flat = t.view(-1)
    idx = torch.randint(0, flat.numel(), (flat.numel() // 16,), generator=g)
    flat[idx] = _SPECIAL[torch.randint(0, len(_SPECIAL), (idx.numel(),), generator=g)]
    return t.to(torch.bfloat16)


@pytest.mark.parametrize("case", list(BITS))
def test_bits_bng_block(device, case):
    shape, (gy, gx), ops = BITS[case]
    mem = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.BLOCK)
    a = ttnn.from_torch(_operand(shape, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b = ttnn.from_torch(_operand(shape, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    for op in ops:
        if op == "mul":
            out = ttnn.multiply(a, b, fast_and_approximate_mode=True, memory_config=mem)
        elif op == "add":
            out = ttnn.add(a, b, memory_config=mem)
        else:
            out = ttnn.subtract(a, b, memory_config=mem)
        ttnn.to_torch(out)
        ttnn.deallocate(out)


# ---------------------------------------------------------------- bits, in-process repeat (determinism) and cross-variant hashes

DET = {
    "geglu_4096x2560_mul": ((1, 1, 4096, 2560), "mul"),
    "geglu_1024x5120_mul": ((1, 1, 1024, 5120), "mul"),
    "geglu_1024x6144_mul": ((1, 1, 1024, 6144), "mul"),
    "tadd_1024x1280_add": ((1, 1, 1024, 1280), "add"),
    "resadd_16384x320_add": ((1, 1, 16384, 320), "add"),
    "tadd_1024x1280_sub": ((1, 1, 1024, 1280), "sub"),
}


def _plain_operand(shape, seed, special):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g) * 4
    if special:
        flat = t.view(-1)
        idx = torch.randint(0, flat.numel(), (flat.numel() // 16,), generator=g)
        flat[idx] = _SPECIAL[torch.randint(0, len(_SPECIAL), (idx.numel(),), generator=g)]
    return t.to(torch.bfloat16)


@pytest.mark.parametrize("special", [False, True], ids=["normal", "special"])
@pytest.mark.parametrize("case", list(DET))
def test_det(device, case, special):
    import hashlib, json

    shape, op = DET[case]
    if shape[-1] % 10 or (shape[-1] // 10) % 32:
        grid = (8, 8)
    else:
        grid = (8, 10)
    mem = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]), strategy=ttnn.ShardStrategy.BLOCK)
    ta, tb = _plain_operand(shape, 1, special), _plain_operand(shape, 2, special)
    outs = []
    for _ in range(3):
        a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        if op == "mul":
            o = ttnn.multiply(a, b, fast_and_approximate_mode=True, memory_config=mem)
        elif op == "add":
            o = ttnn.add(a, b, memory_config=mem)
        else:
            o = ttnn.subtract(a, b, memory_config=mem)
        outs.append(ttnn.to_torch(o).contiguous().view(torch.int16).clone())
        for t in (a, b, o):
            ttnn.deallocate(t)
    rec = {
        "case": f"{case}-{'special' if special else 'normal'}",
        "hashes": [hashlib.sha1(x.numpy().tobytes()).hexdigest()[:16] for x in outs],
        "in_process_diffs": [int((outs[0] != x).sum()) for x in outs[1:]],
    }
    with open(os.environ.get("EB_DET_OUT", "/tmp/eb_det.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
