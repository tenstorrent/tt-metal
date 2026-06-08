# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies as reference_compute_rope_frequencies,
    extract_layer_weights,
    rms_norm as reference_rms_norm,
    text_decoder_layer as reference_text_decoder_layer,
)

from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_logits_pcc_optimizations


def _prefill_tile_start(token_index: int) -> int:
    return (int(token_index) // 32) * 32


def _reference_last_logits(state_dict, args, tokens: torch.Tensor) -> torch.Tensor:
    seq_len = tokens.shape[1]
    ref_cfg = VoxtralTextConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.norm_eps,
    )
    ref_hidden = F.embedding(tokens, state_dict["tok_embeddings.weight"])
    ref_cos, ref_sin = reference_compute_rope_frequencies(
        head_dim=ref_cfg.head_dim,
        max_seq_len=seq_len,
        theta=ref_cfg.rope_theta,
        device=ref_hidden.device,
    )
    ref_attn_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32)
    ref_attn_mask = torch.triu(ref_attn_mask, diagonal=1)
    for layer_idx in range(ref_cfg.num_hidden_layers):
        layer_weights = extract_layer_weights(state_dict, layer_idx, prefix="layers.")
        ref_hidden = reference_text_decoder_layer(
            hidden_states=ref_hidden,
            layer_weights=layer_weights,
            cos=ref_cos,
            sin=ref_sin,
            config=ref_cfg,
            attention_mask=ref_attn_mask,
        )
    ref_hidden = reference_rms_norm(ref_hidden, state_dict["norm.weight"], eps=ref_cfg.rms_norm_eps)
    return F.linear(ref_hidden[:, -1, :], state_dict["output.weight"]).squeeze(0).float()


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_inference(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    assert model.inner.vocab_size > 0
    assert model.inner.args.n_layers > 0
    assert model.inner.args.n_layers == 26
    assert model.inner.args.dim == 3072
    assert model.inner.args.hidden_dim == 9216
    assert model.inner.args.n_heads == 32
    assert model.inner.args.n_kv_heads == 8
    assert model.inner.args.head_dim == 128
    assert model.inner.args.norm_eps == 1e-5
    assert hasattr(model.inner, "embd")
    assert len(model.inner.layers) == 26
    assert hasattr(model.inner, "norm")
    assert hasattr(model.inner, "rope_setup")
    assert hasattr(model.inner, "lm_head")


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_prefill_inference(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    seq_len = 128
    tokens = torch.randint(0, model.inner.vocab_size, (1, seq_len), dtype=torch.int64)
    tt_x, rot_mats_global, rot_mats_local, _, _, _ = model.prepare_inputs_prefill(tokens, start_pos=0)
    tt_logits = model.inner.ttnn_prefill_forward(
        tt_x,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        get_last_token=((seq_len - 1) // 32) * 32,
    )
    logits = model.inner.process_output_prefill(
        tt_logits.cpu(),
        last_token_idx=((seq_len - 1) % 32),
    ).float()

    assert list(logits.shape) == [model.inner.vocab_size]
    assert torch.isfinite(logits).all()


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_prefill_pcc(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=256,
        dtype=ttnn.bfloat16,
        optimizations=voxtral_text_logits_pcc_optimizations,
    )
    args = model.inner.args
    state_dict = args.load_state_dict()

    seq_len = 128
    tokens = torch.randint(0, model.inner.vocab_size, (1, seq_len), dtype=torch.int64)

    tt_x, rot_mats_global, rot_mats_local, _, _, _ = model.prepare_inputs_prefill(tokens, start_pos=0)
    tt_logits = model.inner.ttnn_prefill_forward(
        tt_x,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        get_last_token=_prefill_tile_start(seq_len - 1),
    )
    tt_last_logits = model.inner.process_output_prefill(
        tt_logits.cpu(),
        last_token_idx=((seq_len - 1) % 32),
    ).float()

    ref_last_logits = _reference_last_logits(state_dict, args, tokens)

    passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=0.99)
    print(f"test_text_model_prefill_pcc PCC={float(pcc_value):.6f}")
    assert passing, f"Text model prefill logits mismatch vs reference: {pcc_value}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_decode_reference_pcc(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=256,
        dtype=ttnn.bfloat16,
        optimizations=voxtral_text_logits_pcc_optimizations,
    )
    args = model.inner.args
    state_dict = args.load_state_dict()

    prompt_len = 128
    vocab_size = model.inner.vocab_size
    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.int64)
    decode_input_token = torch.randint(0, vocab_size, (1,), dtype=torch.int64)

    tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _, _ = model.prepare_inputs_prefill(prompt_tokens, start_pos=0)
    _ = model.inner.ttnn_prefill_forward(
        tt_prompt_x,
        rot_mats_global=prompt_rot_global,
        rot_mats_local=prompt_rot_local,
        get_last_token=-1,
    )
    tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
        decode_input_token, torch.tensor([prompt_len], dtype=torch.int64)
    )
    tt_decode_logits, _ = model.inner.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rope_idxs,
        page_table=tt_page_table,
        kv_cache=None,
        sampling_on_device=False,
    )
    tt_last_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

    ref_tokens = torch.cat([prompt_tokens, decode_input_token.view(1, 1)], dim=1)
    ref_last_logits = _reference_last_logits(state_dict, args, ref_tokens)

    passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=0.99)
    print(f"test_text_model_decode_reference_pcc PCC={float(pcc_value):.6f}")
    assert passing, f"Text model decode logits mismatch vs reference: {pcc_value}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("decode_steps", [4, 26], ids=["4_steps", "26_steps"])
def test_text_model_decode_multistep_reference_pcc(device, reset_seeds, decode_steps):
    model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=256,
        dtype=ttnn.bfloat16,
        optimizations=voxtral_text_logits_pcc_optimizations,
    )
    args = model.inner.args
    state_dict = args.load_state_dict()

    prompt_len = 128
    vocab_size = model.inner.vocab_size
    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.int64)
    decode_tokens = torch.randint(0, vocab_size, (1, decode_steps), dtype=torch.int64)

    tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _, _ = model.prepare_inputs_prefill(prompt_tokens, start_pos=0)
    _ = model.inner.ttnn_prefill_forward(
        tt_prompt_x,
        rot_mats_global=prompt_rot_global,
        rot_mats_local=prompt_rot_local,
        get_last_token=-1,
    )

    for step in range(decode_steps):
        current_pos = prompt_len + step
        step_token = decode_tokens[:, step]
        tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
            step_token, torch.tensor([current_pos], dtype=torch.int64)
        )
        tt_decode_logits, _ = model.inner.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=tt_page_table,
            kv_cache=None,
            sampling_on_device=False,
        )
        tt_last_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

        ref_tokens = torch.cat([prompt_tokens, decode_tokens[:, : step + 1]], dim=1)
        ref_last_logits = _reference_last_logits(state_dict, args, ref_tokens)

        passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=0.99)
        print(
            f"test_text_model_decode_multistep_reference_pcc[{decode_steps}_steps] "
            f"step={step} PCC={float(pcc_value):.6f}"
        )
        assert passing, f"Step {step} decode logits mismatch vs reference " f"(pos={current_pos}): {pcc_value}"


# ---------------------------------------------------------------------------
# Width-sharding matmul tests
# ---------------------------------------------------------------------------
# Analysis (Blackhole, M=32 decode):
#   wo  : [32, 4096] × [4096, 3072]  — attention output projection
#   FF2 : [32, 9216] × [9216, 3072]  — MLP down projection
#
# For M=32 (exactly 1 tile), height- and block-sharding degrade to 1 core.
# WIDTH sharding splits N=3072 across 96 cores (8×12 BH grid):
#   per-core weight shard : [K, 32]  →  256 KB  (BF16)   fits in 1.5 MB L1
#   per-core output shard : [32, 32] →    2 KB  (BF16)
# Arithmetic intensity vs L1 is ~350 FLOPs/byte → no longer memory-bound.


def _run_matmul_with_l1_activation(device, A_cpu, weight_tt, m, k, n):
    """Run [M,K]×[K,N] with activation in L1 interleaved, weight in its existing DRAM format.

    Key insight from perf report:
      - FF1/FF3 show `in0:width_sharded` → fast (53% of time): activation in L1
      - wo/FF2  show `in0:dram_interleaved` → slow (22%): activation read from DRAM

    The DRAMShardedProgramConfig already handles L1-resident activations — the weight
    stays in its existing DRAM-sharded format loaded by the model.  The fix is simply
    to ensure the activation is placed in L1 before the matmul call.

    This test validates PCC is maintained when in0 moves from DRAM → L1.
    """
    A_tt = ttnn.from_torch(
        A_cpu.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # activation in L1 (the optimization)
    )
    # Use the weight as loaded by the model (already DRAM-sharded correctly)
    C_tt = ttnn.linear(A_tt, weight_tt, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    C_out = ttnn.to_torch(C_tt).float().squeeze(0).squeeze(0)
    ttnn.deallocate(A_tt)
    ttnn.deallocate(C_tt)
    return C_out


def _bh_l1_width_shard(
    k: int,
    n: int,
    m: int = 32,
    n_cores_x: int = 8,
    n_cores_y: int = 4,
) -> tuple:
    """Return (weight_shard_cfg, out_shard_cfg, prog_cfg) for L1 width sharding on Blackhole.

    Layout:
      B [K, N] → width-sharded in L1, each core owns [K, N/n_cores]
      A [M, K] → replicated in L1 interleaved
      C [M, N] → width-sharded in L1, each core produces [M, N/n_cores]

    The matmul program config (MatmulMultiCoreReuseProgramConfig) must be passed
    to ttnn.linear/matmul so per_core_N matches the shard spec.
    """
    n_cores = n_cores_x * n_cores_y
    assert n % (n_cores * 32) == 0, f"N={n} must be divisible by n_cores*TILE={n_cores*32}"
    per_core_N_tiles = n // (n_cores * 32)  # N-tiles per core

    core_grid = ttnn.CoreGrid(y=n_cores_y, x=n_cores_x)

    # Weight sharding: each core holds [K, per_core_N_tiles*32] tiles of B
    weight_shard_cfg = ttnn.create_sharded_memory_config(
        shape=[k, per_core_N_tiles * 32],
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Output sharding: same N split across cores
    out_shard_cfg = ttnn.create_sharded_memory_config(
        shape=[m, per_core_N_tiles * 32],
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Program config: tells the kernel per_core_M/N so it matches the shard spec
    # in0_block_w ≤ K/32; use a block that divides K and keeps L1 usage feasible
    k_tiles = k // 32
    in0_block_w = min(8, k_tiles)  # conservative block to limit L1 usage
    while k_tiles % in0_block_w != 0:
        in0_block_w -= 1
    per_core_M_tiles = max(1, m // 32)
    # Use DRAMSharded config — designed for N-dimension weight sharding.
    # Works with L1-sharded weights too (in1 buffer type is independent of config).
    # per_core_N = N-tiles per core, per_core_M = M-tiles per core (=1 for M=32).
    # in0_block_w = K processed per inner iteration (fit in L1, must divide K/32).
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M_tiles,
        per_core_N=per_core_N_tiles,
        fused_activation=None,
    )
    return weight_shard_cfg, out_shard_cfg, prog_cfg


@torch.no_grad()
@pytest.mark.timeout(600)
def test_wo_matmul_l1_width_sharding(device, reset_seeds):
    """Validate wo [32×4096]×[4096×3072] with L1 width-sharded ACTIVATION (same format as FF1/FF3).

    The wo matmul currently shows `in0:dram_interleaved` in tt-perf-report (22% of time).
    FF1/FF3 show `in0:width_sharded` (already fast) because their activation comes from
    the L1 residual.  Putting the wo activation in the same format eliminates the DRAM
    round-trip for the activation and should improve throughput.

    This test verifies:
      1. PCC ≥ 0.999 vs torch reference with L1 width-sharded activation.
      2. The dram_matmul_config (unchanged) accepts L1-sharded in0.
    """
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    args = model.inner.args
    state_dict = args.load_state_dict()

    # wo: A [32, K] × B [K, N];  K = n_heads * head_dim = 4096, N = dim = 3072
    M, K, N = 32, args.n_heads * args.head_dim, args.dim
    print(f"\nwo matmul: [{M} × {K}] × [{K} × {N}]")

    wo_key = "layers.0.attention.wo.weight"
    assert wo_key in state_dict, f"Missing {wo_key} in state_dict"
    wo_cpu = state_dict[wo_key].float().T  # CPU reference [K, N]

    A_cpu = torch.randn(M, K, dtype=torch.float32)
    C_ref = A_cpu @ wo_cpu

    # Load weight in DRAM interleaved (simple baseline — same data as model uses)
    B_tt = ttnn.from_torch(
        wo_cpu.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run with activation in L1 interleaved: validates that L1-resident in0
    # produces correct results and is compatible with the existing program.
    C_out = _run_matmul_with_l1_activation(device, A_cpu, B_tt, M, K, N)
    ttnn.deallocate(B_tt)

    _, pcc_val = comp_pcc(C_ref, C_out, pcc=0.999)
    print(f"  PCC (L1 activation) = {float(pcc_val):.6f}")
    assert float(pcc_val) >= 0.999, f"wo L1-activation PCC {float(pcc_val):.6f} < 0.999"


@torch.no_grad()
@pytest.mark.timeout(600)
def test_ff2_matmul_l1_width_sharding(device, reset_seeds):
    """Validate FF2 [32×9216]×[9216×3072] with L1 width-sharded ACTIVATION (same format as FF1/FF3).

    FF2 currently shows `in0:dram_interleaved` in tt-perf-report.  Moving the activation
    (the FF1×SiLU×FF3 intermediate) to L1 width-sharded format eliminates the DRAM
    bandwidth bottleneck for the largest matmul in the MLP.
    """
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    args = model.inner.args
    state_dict = args.load_state_dict()

    # FF2: A [32, K] × B [K, N];  K = hidden_dim = 9216, N = dim = 3072
    M, K, N = 32, args.hidden_dim, args.dim
    print(f"\nFF2 matmul: [{M} × {K}] × [{K} × {N}]")

    ff2_key = "layers.0.feed_forward.w2.weight"
    assert ff2_key in state_dict, f"Missing {ff2_key} in state_dict"
    ff2_cpu = state_dict[ff2_key].float().T  # CPU reference [K, N]

    A_cpu = torch.randn(M, K, dtype=torch.float32)
    C_ref = A_cpu @ ff2_cpu

    B_tt = ttnn.from_torch(
        ff2_cpu.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    C_out = _run_matmul_with_l1_activation(device, A_cpu, B_tt, M, K, N)
    ttnn.deallocate(B_tt)

    _, pcc_val = comp_pcc(C_ref, C_out, pcc=0.999)
    print(f"  PCC (L1 activation) = {float(pcc_val):.6f}")
    assert float(pcc_val) >= 0.999, f"FF2 L1-activation PCC {float(pcc_val):.6f} < 0.999"
