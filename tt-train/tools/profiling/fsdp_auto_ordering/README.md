# FSDP step with derived ordering (experiment)

A tt-train FSDP Llama step (4 blocks, hidden 2048, sequence 512, batch 8, AdamW) on a 1x2 Blackhole mesh. Collectives
run on command queue 1 on the last column of cores, and compute runs on queue 0 on the other columns. ttml's FSDP hooks
are replaced by a prefetch with no waits, and the TTNN ordering ledger derives every wait at launch.

Not for merge. The experiment relies on shortcuts:

- `experimental::SetReservedComputeColumns` narrows the compute grid every mesh device reports;
- `TTML_NUM_COMMAND_QUEUES=2` opens ttml's device with two queues;
- `TTNN_QUEUE_ORDERING_SKIP_REUSE=1` leaves memory reuse unordered, as an unsafe diagnostic.

| File | What |
| --- | --- |
| `exp_fsdp_step.py` | The step and its variants (`--variants`), timing, losses and parameter checksums |
| `final2_h2048_l4.json` | Three interleaved passes of every variant; the published chart's data |
| `final_h2048_l4.json`, `split*_h2048_l4.json` | Earlier passes: components (`noshard`, `ccl0`, `ccl1`) and diagnostics |
| `census_*.log`, `census_*.json` | Edge logs of one step (`TTNN_QUEUE_ORDERING_LOG`); the JSON's `log_marks` are byte offsets at step ends |
| `derived_ordering_fsdp.html`, `make_chart.py` | The write-up and the script that fills its chart from a results file |

```
ulimit -Su $(ulimit -Hu); ./build_metal.sh -e --build-tt-train
V=serial,overlap,eager,eagerpark,marks,dataonly,racy
python tt-train/tools/profiling/fsdp_auto_ordering/exp_fsdp_step.py --steps 6 --warmup 2 \
    --variants $V,$V,$V --init init.pt --out final2_h2048_l4.json
```

`--init` saves the initial weights on the first run and loads them afterwards, since ttml's initialization is not
reproducible under `set_seed`.
