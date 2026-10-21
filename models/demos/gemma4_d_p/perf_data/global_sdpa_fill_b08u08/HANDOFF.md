# Handoff: Gemma4 global ring SDPA experiments (from bh-glx-120-b08u08, 2026-10-21)

Results and analysis: `SUMMARY.md` (this folder). Raw log of every run: `results.txt`.

## Branches (pushed to origin, all on main `909748641c1`, no PRs)

| Branch | Tip | Status |
|---|---|---|
| `mmanzoor/gemma4-gsdpa-ksplit8-q128` | `d18cb26ad6c` | Best: 256k/2K 10310.8 -> 9906.1 ms (-3.9%), accuracy OK. 4K/8K e2e not run yet |
| `mmanzoor/gemma4-gsdpa-v-from-k` | `2e8c27c99a7` | 256k/2K -2.8%, bit-exact. 4K/8K not run |
| `mmanzoor/gemma4-gsdpa-v-from-k-overlap` | `591414c1684` | Negative result (on top of v-from-k) |
| `mmanzoor/gemma4-gsdpa-fill-data` | data only | `[DO NOT MERGE]` copy of this folder's small files under `models/demos/gemma4_d_p/perf_data/global_sdpa_fill_b08u08/` |

`ksplit8-q128` changes host code (K-split cap in the program factory): `ninja -C build && ninja -C build install`
after checking it out. The other two are kernel-only (JIT), but the runner rebuilds on every branch switch anyway.

## Resume on a new machine

```bash
cd ~/compiler/tt-metal && git fetch origin
git fetch origin mmanzoor/gemma4-gsdpa-ksplit8-q128 mmanzoor/gemma4-gsdpa-v-from-k \
    mmanzoor/gemma4-gsdpa-v-from-k-overlap mmanzoor/gemma4-gsdpa-fill-data
for b in ksplit8-q128 v-from-k v-from-k-overlap; do git branch -f mmanzoor/gemma4-gsdpa-$b origin/mmanzoor/gemma4-gsdpa-$b; done
```

Runner (`runner.sh`): queue lines `tag|kind|arg|branch|patch|ENV=a ENV=b`, kinds `layer` (tracy layer benchmark at
chunk_idx `arg`, 2K global layer), `e2e` (traced 256k prefill, chunk size `arg`), `acc` (SDPA nightly test file,
`-k arg`), `kvpcc` (test_prefill_migration[mock-256k], chunk size `arg`, Weka paths built in), `script`. It checks
out the branch (tree must be clean) and rebuilds, applies/reverts the patch, logs to `results.txt`, and stops on a
timeout (likely device hang) or a `STOP` line. Before starting:

1. Copy the folder (or use it in place on /data) and edit `W=` and `REPO=` at the top of `runner.sh`.
2. `cd $REPO && source activate` must work (sets TT_METAL_CACHE etc.; see the repo-root `activate`).
3. Check nobody else is using the galaxy (`ps aux | grep -E "pytest|tt-smi|tt-device"`); b08u08 was reset by
   another user mid-run.
4. Start in tmux: `tmux new-session -d -s gsdpa "$W/runner.sh"`; watch `results.txt`.

Accuracy selector used (7 cases on main, 9 on ksplit8-q128):
`test_ring_joint_attention_gemma4_global_ksplit_accuracy and packed_kv or test_ring_joint_attention_gemma4_global_lofi_matmul_accuracy or test_ring_joint_attention_gemma4_global_segmented_accuracy`
(not `-k gemma4_global`: that also selects the long chunked perf sweep).

Commits on b08u08 used `fmt_commit.sh` (black -l 120 + clang-format, then `--no-verify`) because pre-commit
could not install its environments there. Use normal pre-commit where it works.

## Next experiments (queue lines ready in `queue_next.txt`)

1. 4K / 8K e2e for both branches (cut off by the reset), plus main again on the new box.
2. Depth curve (`layer ci`) and chunk-64 layer for main and ksplit8-q128 on the new box, to re-anchor numbers.
3. Partial readiness on top of ksplit8-q128 (first-shard stall ~150 us / 6% at chunk 64): the port
   (`patches/` of /data/mmanzoor/global_sdpa_gap_c04u02, branch `mmanzoor/gemma4-global-sdpa-gap`) deadlocks on
   main; debug with DPRINT of the reader's semaphore thresholds, under `timeout`, alone in a job.
4. Lazy rescale with in-place accumulation (up to ~10% of global SDPA on ksplit8-q128).
