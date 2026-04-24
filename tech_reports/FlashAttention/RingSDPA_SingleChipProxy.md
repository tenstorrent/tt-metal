# Single-Chip SDPA Proxy for Ring-Joint Per-Iter Perf

Adds a single-chip SDPA mode that reproduces the per-device, per-ring-iter
work of multi-chip `ring_joint_scaled_dot_product_attention`, so per-iter
perf can be tuned and profiled without a multi-chip harness.

## Background

Ring joint SDPA is a multi-device causal+balanced SDPA. Each device runs
`ring_size` iterations; on iter `R`, every device attends its local Q against
K/V rotated from device `(self - R) mod ring_size`.

- **Iter 0 ("diag")** — causal; local Q × local K/V with full causal masking.
- **Iter > 0** — non-causal and zigzag-balanced so every device does one of:
  - **UP** — full Q, half K. All Q chunks assigned; each walks `k_num_chunks/2`.
  - **DOWN** — half Q (heavy half), full K. Only `q_num_chunks/2` Q slots.

## What this branch added

All changes are additive and opt-in; default behavior of existing SDPA
callers is unchanged.

### SDPAProgramConfig knobs

- `flatten_work: bool` — remap work across cores via a shared internal helper
  with CT-arg zigzag, matching `ring_joint_sdpa`'s flat distribution.
- `ring_proxy_case ∈ {none, up, down}` — skip half-iter work the same way
  the ring kernel does for non-causal iters. Keeps full-size tensors
  (`Q[L], K[L], V[L]`) on device so DRAM layout, tile addresses, and core
  dispatch match the ring case exactly.

### KV chain forwarding (UP/DOWN proxy)

L1→L1 store-and-forward of K/V across cores that share a (batch, head) —
the dominant throughput lever when `Sq_chunk_t = Sk_chunk_t` is small and
`DHt` is large (MLA 100k regime). Wired into the `flatten_work && !is_causal`
path by porting `ring_joint_sdpa`'s flat-chain topology build into the
single-chip factory and widening the reader's chain branches to fire in
`SDPA_FLAT_WORK` slots.

### KV chain forwarding (causal flat-work, "A-narrow")

Extended chain forwarding to `flatten_work && is_causal` — the iter-0 proxy.
Chain-participating cores loop the full `k_num_chunks` regardless of Q
position, and compute's existing lightweight causal mask zeroes columns past
each Q's true `q_high`. Hierarchical causal SDPA (`flatten_work=false`) keeps
its tuned non-chain path via the predicate:

    chain_enabled = !is_chunked && (flatten_work || !is_causal)

### Other flat-work changes

- Linear chain order with no descending-q injector reorder — matches
  ring_joint's natural core-idx → phys_x spread across DRAM channels.
- Skip past-diagonal K chunks on the compute side for causal flat-work
  (early exit once `k_chunk > q_high / Sk_chunk_t`).
- Skip the mid-barrier for K/V reads in the single-chip SDPA reader.
- Pass `use_lightweight_causal_mask` through to `sdpa_standard` so flat-work
  causal doesn't hang on the padded region.

### Test & measurement hooks

- Adds `mla_100k_ring_iter_{0,up,down}` rows in
  `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`.
  Head-dim-V is inferred from V shape so MLA's asymmetric head dims work.
- Per-model `flatten_work` wiring in the sprint test; WAN suites restored.

The sibling `TT_METAL_RING_ITER_ONLY=R` env-var hook on
`test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table`
(runs only ring iter `R`, skips AllGather, reports per-iter math util)
ships in a separate branch/PR so this branch stays scoped to the
single-chip proxy; the two are used together for the per-iter comparison
in the table below.

## Current numbers (MLA 100k, Blackhole, QB)

Single-chip proxy, `nhq=32`, 110 cores, `q160/k160`:

| Proxy            | Duration (ms) | Math util |
|------------------|---------------|-----------|
| `ring_iter_0`    | 2.435         | 31.1%     |
| `ring_iter_up`   | 1.417         | 53.5%     |
| `ring_iter_down` | 1.332         | 56.9%     |

Multi-chip ring-joint per-device per-iter baseline (`nhq=29`, 80 cores,
`ring_size=4`, via `TT_METAL_RING_ITER_ONLY`) lands UP at 52.4–53.0% and
DOWN at 56.0–56.5% — UP/DOWN proxies track ring within noise. Iter-0 causal
ring is 30.8–31.4%; the proxy now sits inside that band, so the A-narrow
over-read cost (chain participants walk full `k_num_chunks` and rely on the
lightweight causal mask to zero over-read columns) is absorbed by the
parameter mismatch rather than showing up as a standalone gap.

## Usage

### Running the single-chip proxy tests

The sprint file
(`tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`)
parametrizes over model configs including `mla_100k_ring_iter_{0,up,down}`
and `mla_128k_ring_iter_0`. Example:

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_down-q160-k160]
```

`run_safe_pytest.sh` serializes device access, appends `-x`, and resets the
device after the run.

### Per-iter ring-joint baseline

```bash
TT_METAL_RING_ITER_ONLY=0 scripts/run_safe_pytest.sh \
  tests/.../test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table
```

Set `TT_METAL_RING_ITER_ONLY` to `0` for causal/diag, or to a non-zero iter
to measure a non-causal UP/DOWN iter in isolation.
