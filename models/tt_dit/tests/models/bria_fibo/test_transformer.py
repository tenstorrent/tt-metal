# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.tt_dit.blocks.attention import Attention
from models.tt_dit.blocks.transformer_block import TransformerBlock
from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTextProjection, inject_text
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as tt_tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_ref_transformer(dtype=torch.bfloat16):
    try:
        from diffusers import BriaFiboTransformer2DModel
    except Exception:
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
    try:
        # When running offline, resolve the HF repo ID to its local cache path.
        fibo_path = FIBO_PATH
        if not os.path.isdir(fibo_path):
            from huggingface_hub import snapshot_download

            fibo_path = snapshot_download(fibo_path, allow_patterns=["transformer/*"], local_files_only=True)
        return BriaFiboTransformer2DModel.from_pretrained(fibo_path, subfolder="transformer", torch_dtype=dtype).eval()
    except Exception as e:
        pytest.skip(f"FIBO transformer unavailable: {e}")


def test_fibo_transformer_reference_config():
    m = _load_ref_transformer()
    c = m.config
    assert c.num_layers == 8 and c.num_single_layers == 38
    assert c.num_attention_heads == 24 and c.attention_head_dim == 128
    assert c.in_channels == 48 and c.joint_attention_dim == 4096
    assert c.axes_dims_rope == [16, 56, 56]
    assert len(m.caption_projection) == c.num_layers + c.num_single_layers  # 46


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_text_projection(*, mesh_device):
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTextProjection
    from models.tt_dit.utils import tensor as tt_tensor
    from models.tt_dit.utils.check import assert_quality

    m = _load_ref_transformer()
    ref = m.caption_projection[0]  # HF BriaFiboTextProjection
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2048)
    with torch.no_grad():
        r = ref(x.to(torch.bfloat16))
    tt = BriaFiboTextProjection(in_features=2048, hidden_size=1536, mesh_device=mesh_device)
    tt.load_torch_state_dict(ref.state_dict())
    out = tt.forward(tt_tensor.from_torch(x.to(torch.bfloat16), device=mesh_device))
    assert tuple(tt_tensor.to_torch(out).shape)[-1] == 1536
    assert_quality(r, tt_tensor.to_torch(out), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_timestep_embed(*, mesh_device):
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTimestepEmbed
    from models.tt_dit.utils import tensor as tt_tensor
    from models.tt_dit.utils.check import assert_quality

    m = _load_ref_transformer()
    ref = m.time_embed  # HF BriaFiboTimestepProjEmbeddings
    inner_dim = m.config.num_attention_heads * m.config.attention_head_dim  # 3072
    torch.manual_seed(0)
    timestep = torch.tensor([500.0, 250.0])
    with torch.no_grad():
        r = ref(timestep, dtype=torch.bfloat16)
    tt = BriaFiboTimestepEmbed(inner_dim=inner_dim, mesh_device=mesh_device)
    tt.load_torch_state_dict(ref.state_dict())
    # Pass timestep as [batch, 1] bfloat16 tensor on device
    tt_timestep = tt_tensor.from_torch(timestep.unsqueeze(-1).to(torch.bfloat16), device=mesh_device)
    out = tt.forward(tt_timestep)
    out_torch = tt_tensor.to_torch(out)
    assert tuple(out_torch.shape)[-1] == inner_dim
    assert_quality(r, out_torch, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_dual_block(*, mesh_device):
    """Dual (double) transformer block with concat-halves injection at tp=1.

    Validates that:
      1. inject_text(ctx, projected) produces the correct concat-halves on device.
      2. tt_dit's TransformerBlock (context_pre_only=False), loaded from the FIBO
         dual-block weights, reproduces both output streams (spatial + context)
         relative to the HF reference block with PCC >= 0.99.

    Reference return order: BriaFiboTransformerBlock.forward returns
      (encoder_hidden_states, hidden_states), i.e. context first, spatial second.
    TT TransformerBlock.forward returns (spatial, prompt), i.e. spatial first.
    """
    m = _load_ref_transformer()
    ref_block = m.transformer_blocks[0]
    ref_proj = m.caption_projection[0]

    inner_dim = m.config.num_attention_heads * m.config.attention_head_dim  # 3072
    num_heads = m.config.num_attention_heads  # 24
    head_dim = m.config.attention_head_dim  # 128
    half_dim = inner_dim // 2  # 1536
    text_encoder_dim = 2048

    # Sequence lengths: small enough to run quickly
    batch_size = 1
    spatial_seq_len = 256
    prompt_seq_len = 64

    torch.manual_seed(42)
    spatial = torch.randn(batch_size, spatial_seq_len, inner_dim, dtype=torch.bfloat16)
    prompt = torch.randn(batch_size, prompt_seq_len, inner_dim, dtype=torch.bfloat16)
    text_layer = torch.randn(batch_size, prompt_seq_len, text_encoder_dim, dtype=torch.bfloat16)
    temb = torch.randn(batch_size, inner_dim, dtype=torch.bfloat16)
    total_seq_len = prompt_seq_len + spatial_seq_len
    rope_cos = torch.randn(total_seq_len, head_dim, dtype=torch.bfloat16)
    rope_sin = torch.randn(total_seq_len, head_dim, dtype=torch.bfloat16)

    # --- Reference forward (torch) ---
    with torch.no_grad():
        projected = ref_proj(text_layer)  # [batch, P, 1536]
        # Inject: replace upper half of context features
        ctx_injected = torch.cat([prompt[:, :, :half_dim], projected], dim=-1)  # [batch, P, 3072]
        # ref_block returns (encoder_hidden_states, hidden_states) = (context, spatial)
        ref_ctx_out, ref_spatial_out = ref_block(
            hidden_states=spatial,
            encoder_hidden_states=ctx_injected,
            temb=temb,
            image_rotary_emb=(rope_cos, rope_sin),
        )

    # --- TT setup (tp=1, sp=1 at mesh (1,1)) ---
    sp_axis = 0
    tp_axis = 1
    sp_factor = 1
    tp_factor = 1

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    padding_config = None  # 24 heads % 1 == 0

    tt_block = TransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_pre_only=False,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_block.load_torch_state_dict(ref_block.state_dict())

    # Build TT tensors (tp=1: no sharding needed, but use same API as production)
    spatial_padded = Attention.pad_spatial_sequence(spatial, sp_factor=sp_factor)
    spatial_rope_cos_padded = Attention.pad_spatial_sequence(rope_cos[prompt_seq_len:], sp_factor=sp_factor)
    spatial_rope_sin_padded = Attention.pad_spatial_sequence(rope_sin[prompt_seq_len:], sp_factor=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_prompt = bf16_tensor(prompt, device=mesh_device, mesh_axis=tp_axis, shard_dim=2)
    tt_time_embed = bf16_tensor(temb.unsqueeze(1), device=mesh_device)
    tt_spatial_rope_cos = bf16_tensor(spatial_rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(spatial_rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device)

    # Inject text: project text_layer on device, then inject into context
    tt_text_layer = bf16_tensor(
        text_layer, device=mesh_device
    )  # replicated (projected to half_dim before any sharding)
    tt_proj = BriaFiboTextProjection(in_features=text_encoder_dim, hidden_size=half_dim, mesh_device=mesh_device)
    tt_proj.load_torch_state_dict(ref_proj.state_dict())
    tt_projected = tt_proj.forward(tt_text_layer)

    tt_prompt_injected = inject_text(tt_prompt, tt_projected)

    # --- TT forward ---
    tt_spatial_out, tt_prompt_out = tt_block.forward(
        tt_spatial,
        tt_prompt_injected,
        tt_time_embed,
        spatial_sequence_length=spatial_seq_len,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
    )

    ttnn.synchronize_device(mesh_device)

    # Retrieve outputs
    tt_spatial_torch = tt_tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])
    tt_spatial_torch = tt_spatial_torch[:, :spatial_seq_len]
    tt_prompt_torch = tt_tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])

    # Assert PCC >= 0.99 on both streams
    assert_quality(ref_spatial_out, tt_spatial_torch, pcc=0.99)
    assert_quality(ref_ctx_out, tt_prompt_torch, pcc=0.99)


def _truncate_ref(ref, num_layers: int, num_single_layers: int):
    """Truncate the reference's block ModuleLists in place for reduced-depth iteration.

    ``caption_projection`` is indexed by a running ``block_id`` spanning both loops, so keep
    only its first ``num_layers + num_single_layers`` entries.
    """
    import torch.nn as nn

    ref.transformer_blocks = nn.ModuleList(list(ref.transformer_blocks)[:num_layers])
    ref.single_transformer_blocks = nn.ModuleList(list(ref.single_transformer_blocks)[:num_single_layers])
    ref.caption_projection = nn.ModuleList(list(ref.caption_projection)[: num_layers + num_single_layers])
    ref.config.num_layers = num_layers
    ref.config.num_single_layers = num_single_layers
    return ref


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_fibo_transformer(*, mesh_device):
    """Full ``BriaFiboTransformer`` at tp=1 vs the HF reference, PCC >= 0.99.

    Reduced depth via ``FIBO_DUAL`` / ``FIBO_SINGLE`` env knobs (default 2/2). Set both to
    the full config (8/38) to run the whole model.
    """
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboCheckpoint

    num_dual = int(os.environ.get("FIBO_DUAL", "2"))
    num_single = int(os.environ.get("FIBO_SINGLE", "2"))

    # --- Reference (truncated to reduced depth) ---
    ref = _load_ref_transformer()
    ref = _truncate_ref(ref, num_dual, num_single)

    c = ref.config
    in_channels = c.in_channels  # 48
    joint_attention_dim = c.joint_attention_dim  # 4096
    text_encoder_dim = c.text_encoder_dim  # 2048
    num_blocks = num_dual + num_single

    batch_size = 1
    spatial_seq_len = 256
    prompt_seq_len = 64

    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, in_channels]).to(torch.bfloat16)
    prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim]).to(torch.bfloat16)
    # 46-entry list (only first num_blocks are indexed); SAME list to ref and tt.
    text_encoder_layers = [
        torch.randn([batch_size, prompt_seq_len, text_encoder_dim]).to(torch.bfloat16)
        for _ in range(c.num_layers + c.num_single_layers)
    ]
    timestep = torch.full([batch_size], fill_value=500).to(torch.bfloat16)

    # RoPE ids (txt = zeros, img = random), Flux-style.
    text_ids = torch.zeros([prompt_seq_len, 3]).to(torch.bfloat16)
    image_ids = torch.randint(1024 * 1024, [spatial_seq_len, 3]).to(torch.bfloat16)
    ids = torch.cat((text_ids, image_ids), dim=0).to(torch.bfloat16)
    rope_cos, rope_sin = ref.pos_embed.forward(ids)

    # --- Reference forward (RAW timestep, no /1000) ---
    with torch.no_grad():
        ref_out = ref.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            text_encoder_layers=text_encoder_layers,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=text_ids,
        ).sample

    # --- TT setup (tp=1, sp=1 at mesh (1,1)) ---
    sp_axis = 0
    tp_axis = 1
    sp_factor = 1
    tp_factor = 1

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    checkpoint = BriaFiboCheckpoint(FIBO_PATH)
    tt_model = checkpoint.build(
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_layers=num_dual,
        num_single_layers=num_single,
    )

    # --- TT inputs (tp=1 sharding) ---
    spatial_padded = Attention.pad_spatial_sequence(spatial, sp_factor=sp_factor)
    spatial_rope_cos_padded = Attention.pad_spatial_sequence(rope_cos[prompt_seq_len:], sp_factor=sp_factor)
    spatial_rope_sin_padded = Attention.pad_spatial_sequence(rope_sin[prompt_seq_len:], sp_factor=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_prompt = bf16_tensor(prompt, device=mesh_device)
    tt_timestep = bf16_tensor(timestep.unsqueeze(-1), device=mesh_device)
    tt_text_encoder_layers = [bf16_tensor(t, device=mesh_device) for t in text_encoder_layers[:num_blocks]]

    tt_spatial_rope_cos = bf16_tensor(spatial_rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(spatial_rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device)

    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        text_encoder_layers=tt_text_encoder_layers,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )
    ttnn.synchronize_device(mesh_device)

    tt_output_torch = tt_tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, tp_axis])
    tt_output_torch = tt_output_torch[:, :spatial_seq_len]

    assert_quality(ref_out, tt_output_torch, pcc=0.99)
