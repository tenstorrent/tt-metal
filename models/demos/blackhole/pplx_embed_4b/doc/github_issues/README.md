# Kernel / op help requests — drafts for tenstorrent/tt-metal issues

Five issues drafted from the Qwen3-4B (same architecture as the embedding model: dim 2560, 36
layers, 32 Q / 8 KV heads of 128, hidden 9728) Blackhole P150 latency work. Each file carries the
title, the intended assignee, the measurements, the concrete ask and a self-contained repro /
acceptance script that runs on one P150 with `TT_VISIBLE_DEVICES=<id>`.

| file | assignee | ask |
|---|---|---|
| `01_minimal_matmul_wide_n_efficiency.md` | **sankarmanoj-tt** (Sankar Manoj) | why (K=2560, N=9728) runs at 332 TFLOP/s vs 473 for (K=9728, N=2560); ≈ −53 ms at M=16384 |
| `02_minimal_matmul_swiglu_epilogue.md` | **sankarmanoj-tt** | a SwiGLU epilogue that keeps the full accumulation subblock; ≈ −13 ms at M=16384, more with SFPU overlap |
| `03_sdpa_bs1_work_units_and_l1_footprint.md` | **cmaryanTT** (Chris Maryan) | bs1 SDPA fills 64 of 120 cores (78 TFLOP/s); batched SDPA static CBs block L1-resident activations |
| `04_rmsnorm_interleaved_bandwidth.md` | **cmaryanTT** | interleaved rms_norm at ~50% of DRAM bandwidth (two-pass); single-pass or fused add+norm |
| `05_matmul_2d_mcast_dram_sharded_wide_grid.md` | **cmaryanTT** | review/upstream the wide-grid bank-walk fix and the coalesced in1 reads (commit `213530ded2a`) |

Filed 2026-09-24 (bodies in `body_0N.md`, concise: issue / expected / random-data unit test):

| # | issue | assignee |
|---|---|---|
| 01 | https://github.com/tenstorrent/tt-metal/issues/57626 | sankarmanoj-tt |
| 02 | https://github.com/tenstorrent/tt-metal/issues/57627 | sankarmanoj-tt |
| 03 | https://github.com/tenstorrent/tt-metal/issues/57628 | cmaryanTT |
| 04 | https://github.com/tenstorrent/tt-metal/issues/57629 | cmaryanTT |
| 05 | https://github.com/tenstorrent/tt-metal/issues/57630 | cmaryanTT |
| 06 | https://github.com/tenstorrent/tt-metal/issues/57722 (head-major tile-id remap in minimal_matmul; body in `/tmp/body_06.md` → `body_06.md`) | sankarmanoj-tt |
| — | comments on #57627 (SwiGLU product in FF2's in0 path) and #57628 (concat-free SDPA output) from the sustained-clock profiling | — |

To re-create from the drafts (the `0N_*.md` files carry front matter with title/assignee):

```bash
cd models/demos/blackhole/pplx_embed_4b/doc/github_issues
for f in 01 02 03 04 05; do
  file=$(ls ${f}_*.md)
  title=$(grep -m1 '^title:' "$file" | sed 's/^title: *"\(.*\)"$/\1/')
  assignee=$(grep -m1 '^assignee:' "$file" | sed 's/^assignee: *//')
  # body = everything after the front matter
  awk 'f>=2{print} /^---$/{f++}' "$file" > /tmp/body_$f.md
  gh issue create --repo tenstorrent/tt-metal --title "$title" --body-file /tmp/body_$f.md \
     --assignee "$assignee" --label perf --label blackhole
done
```

Handles (from the commits API): Sankar Manoj `sankarmanoj-tt`, Chris Maryan `cmaryanTT`.
