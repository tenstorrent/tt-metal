"""Op-level investigation: can the flow estimator's attention take a real chunk-causal +
padding mask through fused `ttnn.transformer.scaled_dot_product_attention`, is it correct,
and is it faster than the explicit matmul -> scale -> add-bias -> softmax -> matmul chain?

Recreated 2026-09-25 (the 2026-09-23 original was lost with an instance, never committed).
Answers three questions for `tt/flow/decoder.py`'s streaming mode, at the estimator's real
attention shape `[2, 8, T, 64]` (CFG-doubled batch, 8 heads, head_dim 64), bf16, at the real
mel-rate bucket sizes T = 384 / 768 / 1536 (token buckets 192 / 384 / 768, linear step 64, x2):

1. Mask construction on device. SDPA's `attn_mask` must be a full `[1|B, 1|H, Sq, Sk]` bf16
   TILE tensor in DRAM (sdpa_device_operation.cpp validates this; a key-only `[B, 1, 1, Sk]`
   padding row is NOT accepted). Upstream combines the two masks as a logical AND
   (`add_optional_chunk_mask`: `masks & subsequent_chunk_mask(...)`, then
   `mask_to_bias`), whose additive form is `pad_bias[B,1,1,T] + chunk_bias[1,1,T,T]`. Checked
   here that `ttnn.add` broadcasts that sum correctly on device (vs. the same sum on host).
2. Correctness vs. a float64 torch reference, at chunk-ALIGNED valid lengths (multiples of
   CHUNK_SIZE_UP=50: 300, 650, 1500) AND non-aligned ones (310, 660, 1510) -- the non-aligned
   cases are the ones where valid queries in the partial last chunk would see padded keys if
   the padding term were missing. Also reports, for the non-aligned case, what PCC a
   chunk-bias-ONLY mask would give (the bug the padding term prevents).
3. Timing (host wall clock around `synchronize_device`, N reps after warm-up) of:
   SDPA no mask (today's non-streaming path) / SDPA + mask with the production program config
   (q_chunk=128, k_chunk=256) / SDPA + mask with the op's DEFAULT program config (the pitfall
   the lost 09-23 investigation first fell into) / the explicit chain with the same mask.

Run (device; python_env's .pth files put the repo, ttnn/ and tools/ on sys.path, so no PYTHONPATH):
  /home/user/tt-metal/python_env/bin/python .../masked_sdpa_investigation.py
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.flow.decoder import SDPA_K_CHUNK, SDPA_Q_CHUNK
from models.demos.audio.cosyvoice2.tt.flow.encoder import CHUNK_SIZE_UP, chunk_causal_bias_torch

B, H, D = 2, 8, 64
SCALE = D**-0.5
PAD_NEG = -1.0e10  # upstream mask_to_bias
CASES = [(384, 300), (384, 310), (768, 650), (768, 660), (1536, 1500), (1536, 1510)]
REPS = 20


def pad_bias_torch(t_len: int, valid: int) -> torch.Tensor:
    """[B, 1, 1, T]: 0 for valid keys, PAD_NEG for padded ones -- the decoder's existing
    `(mask - 1) * 1e10` padding term, shaped as the decoder builds it."""
    b = torch.zeros(B, 1, 1, t_len)
    b[..., valid:] = PAD_NEG
    return b


def ref_attention(q, k, v, bias):
    q, k, v, bias = (x.double() for x in (q, k, v, bias))
    s = (q @ k.transpose(-2, -1)) * SCALE + bias
    return torch.softmax(s, dim=-1) @ v


def to_dev(x, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def timed(fn, device):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / REPS * 1e3


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        grid = device.compute_with_storage_grid_size()
        prod_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=SDPA_Q_CHUNK,
            k_chunk_size=SDPA_K_CHUNK,
            exp_approx_mode=False,
        )
        rows = []
        for t_len, valid in CASES:
            torch.manual_seed(t_len + valid)
            q, k, v = (torch.randn(B, H, t_len, D) for _ in range(3))
            chunk = chunk_causal_bias_torch(t_len, CHUNK_SIZE_UP)  # [1,1,T,T]
            pad = pad_bias_torch(t_len, valid)  # [B,1,1,T]
            full_bias_host = chunk + pad  # [B,1,T,T]

            q_d, k_d, v_d = to_dev(q, device), to_dev(k, device), to_dev(v, device)
            chunk_d, pad_d = to_dev(chunk, device), to_dev(pad, device)

            # (1) the device-side broadcast sum the decoder will do per estimator call
            mask_d = ttnn.add(chunk_d, pad_d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask_back = ttnn.to_torch(mask_d).float()
            assert tuple(mask_back.shape) == (B, 1, t_len, t_len), mask_back.shape
            # Compare the masked/unmasked PATTERN (values are 0 / ~-3e4 / ~-1e10 / their sum).
            bcast_ok = torch.equal(mask_back < -1.0, full_bias_host < -1.0)

            # (2) correctness at the valid query rows (padded query rows are masked out of the
            # output downstream by the decoder's own `* mask`, as upstream).
            want = ref_attention(q, k, v, full_bias_host)[:, :, :valid]
            got = ttnn.transformer.scaled_dot_product_attention(
                q_d, k_d, v_d, is_causal=False, attn_mask=mask_d, scale=SCALE, program_config=prod_cfg
            )
            got_t = ttnn.to_torch(got).float()[:, :, :valid]
            ttnn.deallocate(got)
            _, pcc_masked = comp_pcc(want, got_t, 0.99)

            chunk_only_d = ttnn.repeat(chunk_d, ttnn.Shape((B, 1, 1, 1)))
            got_co = ttnn.transformer.scaled_dot_product_attention(
                q_d, k_d, v_d, is_causal=False, attn_mask=chunk_only_d, scale=SCALE, program_config=prod_cfg
            )
            _, pcc_chunk_only = comp_pcc(want, ttnn.to_torch(got_co).float()[:, :, :valid], 0.99)
            ttnn.deallocate(got_co)

            # Leak probe: random V makes a few leaked padded keys nearly invisible to PCC (the
            # chunk-only column above stays > 0.99 even when wrong). Put a large value in the
            # PADDED rows of V: any attention weight that reaches a padded key now shows up.
            v_leak = v.clone()
            v_leak[:, :, valid:, :] = 50.0
            v_leak_d = to_dev(v_leak, device)
            want_leak = ref_attention(q, k, v_leak, full_bias_host)[:, :, :valid]
            probe = {}
            for name, m in (("full", mask_d), ("chunk_only", chunk_only_d)):
                o = ttnn.transformer.scaled_dot_product_attention(
                    q_d, k_d, v_leak_d, is_causal=False, attn_mask=m, scale=SCALE, program_config=prod_cfg
                )
                o_t = ttnn.to_torch(o).float()[:, :, :valid]
                ttnn.deallocate(o)
                probe[name] = (comp_pcc(want_leak, o_t, 0.99)[1], (o_t - want_leak.float()).abs().max().item())
            ttnn.deallocate(v_leak_d)
            ttnn.deallocate(chunk_only_d)

            # explicit chain, same mask, same bf16 inputs (K pre-transposed outside the timed
            # region, as the production chain gets it from split_query_key_value_and_split_heads)
            k_t_d = ttnn.transpose(k_d, -2, -1)

            def chain():
                s = ttnn.matmul(q_d, k_t_d)
                s2 = ttnn.multiply(s, SCALE)
                ttnn.deallocate(s)
                s3 = ttnn.add(s2, mask_d)
                ttnn.deallocate(s2)
                a = ttnn.softmax(s3, dim=-1)
                ttnn.deallocate(s3)
                o = ttnn.matmul(a, v_d)
                ttnn.deallocate(a)
                return o

            got_chain = chain()
            _, pcc_chain = comp_pcc(want, ttnn.to_torch(got_chain).float()[:, :, :valid], 0.99)
            ttnn.deallocate(got_chain)

            # (3) timing
            ms_nomask = timed(
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    q_d, k_d, v_d, is_causal=False, scale=SCALE, program_config=prod_cfg
                ),
                device,
            )
            ms_masked = timed(
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    q_d, k_d, v_d, is_causal=False, attn_mask=mask_d, scale=SCALE, program_config=prod_cfg
                ),
                device,
            )
            ms_masked_default = timed(
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    q_d, k_d, v_d, is_causal=False, attn_mask=mask_d, scale=SCALE
                ),
                device,
            )
            ms_chain = timed(chain, device)
            ms_mask_build = timed(lambda: ttnn.add(chunk_d, pad_d, memory_config=ttnn.DRAM_MEMORY_CONFIG), device)

            for t in (q_d, k_d, v_d, k_t_d, chunk_d, pad_d, mask_d):
                ttnn.deallocate(t)
            rows.append(
                (
                    t_len,
                    valid,
                    valid % CHUNK_SIZE_UP == 0,
                    bcast_ok,
                    pcc_masked,
                    pcc_chunk_only,
                    pcc_chain,
                    ms_nomask,
                    ms_masked,
                    ms_masked_default,
                    ms_chain,
                    ms_mask_build,
                    probe,
                )
            )
            print(f"done T={t_len} valid={valid}", flush=True)

        print(
            "\n| T | valid | chunk-aligned | device mask sum == host | PCC masked SDPA | PCC chunk-bias-only SDPA "
            "| PCC explicit chain | ms SDPA no mask | ms SDPA+mask (q128/k256) | ms SDPA+mask (default cfg) "
            "| ms explicit chain | ms mask add | leak probe PCC / max|diff|: full mask | leak probe: chunk-only |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(
                f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.6f} | {r[5]:.6f} | {r[6]:.6f} | {r[7]:.3f} "
                f"| {r[8]:.3f} | {r[9]:.3f} | {r[10]:.3f} | {r[11]:.3f} "
                f"| {r[12]['full'][0]:.6f} / {r[12]['full'][1]:.3g} | {r[12]['chunk_only'][0]:.6f} / {r[12]['chunk_only'][1]:.3g} |"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
