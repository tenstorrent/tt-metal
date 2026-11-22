# Local runner tooling (experiment only)

Set `PUSH_EXP_DIR` (default /tmp/push_exp) to a scratch dir. Create the launcher and ttop dir once:

```
mkdir -p $PUSH_EXP_DIR/ttop $PUSH_EXP_DIR/runs && echo localhost > $PUSH_EXP_DIR/ttop/hostfile
cp models/demos/common/prefill/runners/ci/run_multirank_pcc.sh $PUSH_EXP_DIR/run_local.sh
patch $PUSH_EXP_DIR/run_local.sh < tools/run_local.patch
```

The runner hooks it relies on (per-rank env `SCRATCH_RUNNER_ENV` in run_multirank_pcc.sh; RT dump
`SCRATCH_RT_DUMP`/`SCRATCH_RT_KEEP` in prefill_runner.py) are uncommitted scratch edits; see HANDOFF.md section 0.4.

Example A/B (real weights, depth 8, 1M request):

```
E="export TT_METAL_SHM_TRACKING_DISABLED=1; export SPARSE_MLA_INDEXER_BLOCK_MAX=1;"
tools/drive2.sh glm53 glm_B 8 "$E export SPARSE_MLA_FULL_PUSH=0;"
tools/drive2.sh glm53 glm_C 8 "$E export SPARSE_MLA_FULL_PUSH=1;"
python3 tools/analyze.py glm_B glm_C
```
