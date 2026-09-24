# KV-cache PCC study for #57454: details

Back to the [summary](README.md). The runs are listed in [RUNS.md](RUNS.md).

## Test
- `models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-8k | mock-256k]`
- It prefills a Gutenberg text (Les Misérables) chunk by chunk, reads back all 60 layers' K/V cache, and compares every head
  against a stored HF GPU trace (`/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135`, 262,144 tokens, bf16).
- Gate: the minimum per-head PCC over all layers must be >= 0.91.
- Local additions: a chunk-size override (`GEMMA4_PCC_CHUNK_SIZE`) so one test covers chunk 2048 / 4096 / 8192. Runner:
  `scripts_0923/run_pcc_ctx.sh <label> <chunk> [FLAG=VAL ...]` (`CTX=256k` for long context).
- BH Galaxy 8x4 (CP8 / TP4). One run takes about 5 min at 8k context and 12 min at 256k.

## How to read the table
- **Min PCC @layer**: the gated value. It is always layer 39 (a sliding layer, V head), and it moves in the third
  decimal with any arithmetic change, so pass/fail near 0.91 is close to a coin flip.
- **Err L0-20 vs base**: mean over layers 0-20 of (relative RMSE / base's relative RMSE at the same context and chunk).
  Below 1.0 means more accurate than base. Error grows chaotically with depth (a small early difference is amplified), so
  early layers show an op's own precision best. **L0-20 worse** counts the layers where the run is worse than base.
- **Err L21-59**: the same ratio for the deep layers, which is what decides the gate.
- **Determinism**: base reruns are bit-identical (`base_c4096_r2`, `_r3`, `base_c8192_rep`), so every difference is real.

## Groups
- **A, 8k context** (combined branch with env flags; #57454 as first posted). Base itself FAILS the gate at chunk
  2048 (0.9077) and 4096 (0.8514), so at 8k these runs are compared on the early-layer error only.
- **B, 256k context**, the same flag-based tree: #57454 as first posted, and each change alone.
- **C, 256k context** (mostly chunk 8192, plus M1 at 2048 / 4096), on the perf branch synced to the PR code (`07759d549a6`) with local experiment knobs.

## Findings
1. **#57454 as first posted fails at 256k:** min 0.9094 (chunk 2048) and 0.9083 (8192), against base 0.9106 and 0.9108. Layers
   0-20 are better than base (0.91x), and the deep layers are worse (1.013x to 1.016x).
2. **The MLP explicit blocking causes it** (`mlp256k_c8192`: deep 1.016x, 0.9088). Norm adds a little (1.008x).
   Attention with HiFi2 + fp32 is better than base at every depth (0.9118).
3. **Fidelity is not the cause.** `ttnn.linear` defaults to HiFi2 only when it gets neither a program config nor a
   core grid, and LoFi otherwise. #57454's explicit MLP config therefore moves the MLP from HiFi2 (main) to LoFi, and the
   combined branch's base MLP was already LoFi (core grid). Restoring HiFi2 with bf16 accumulation makes it worse
   (`mlpexp_m3`, 0.9017, deep 1.065x). The earlier "bf16 attention" loss was really this LoFi default.
4. **fp32 accumulation fixes the MLP:** LoFi + fp32 dest (`mlpexp_m1`, 0.9123, 0/21 worse, deep 0.985x).
5. **Cheaper attention does not work at 8192:** M1 + LoFi bf16 attention fails (0.9009), and M1 + HiFi2 bf16 fails (0.9071).
6. **Keeping the MLP on its default path at 8192** passes (`mlpexp_mlpoff`, 0.9118, 0/21 worse, deep 0.993x).
7. **M1 also passes at 2048 and 4096:** 0.9131 (2048: 0/21 worse, deep 0.984x) and 0.9121 (4096; no 256k base run at 4096,
   so no ratio). All three M1 runs clear the gate by more than base does (base 0.9106 / 0.9108).

## Perf for the same configurations (256k, per-chunk cost `a` from the traced demo)
At chunk 8192: #57454 as first posted 196.8 ms; + MLP LoFi fp32 203.8; + MLP HiFi2 bf16 213.5; + MLP HiFi2 fp32 227.8; MLP on its default
path 198.0; base about 204.

| chunk | #57454 as first posted | #57454 + MLP LoFi fp32 (M1) | #57454 with default MLP |
|---|---|---|---|
| 2048 | 92.2 ms / 23.63 s | 92.2 ms / 23.64 s | 103.0 ms / 25.02 s |
| 4096 | 119.6 ms / 13.98 s | 120.1 ms / 14.08 s | 146.8 ms / 15.53 s |
| 8192 | 196.8 ms / 12.63 s | 203.8 ms / 12.84 s | 198.0 ms / 12.66 s |

(per-chunk cost `a` / 256k device total). The explicit MLP blocking is worth 11 ms at 2048 and 27 ms at 4096, and nothing
at 8192 once it accumulates in fp32.

## What #57454 does now
Commit 993720c3d3c: the MLP projections accumulate in fp32, and the explicit MLP blocking is only used for short-M shapes
(per-core M <= 2, i.e. chunk <= 4096 at CP8). At chunk 8192 the MLP keeps its default path.

## Files
- [RUNS.md](RUNS.md): one row per run, then the exact flags and tree sha for each.
- [per_layer.csv](per_layer.csv): per-layer min head PCC, layer PCC and relative RMSE for every run.
- Raw logs (pytest output + runner log per run): https://gist.github.com/kmabeeTT/edde8493f9ff0d889185be0742705821
- Flag names: group A/B `GEMMA4_NORM_SHARD`, `GEMMA4_MLP_MM_CFG` (+`_GRID=12x10`), `GEMMA4_ATTN_MM_PC`,
  `GEMMA4_DIAG_ATTN_FP32ACC` / `_PACKER_L1`, `GEMMA4_DIAG_MLP_FP32ACC` (HiFi2 + fp32). Group C `GEMMA4_EXP_MLP_*`,
  `GEMMA4_EXP_ATTN`. All are local experiment switches, not part of any PR.
