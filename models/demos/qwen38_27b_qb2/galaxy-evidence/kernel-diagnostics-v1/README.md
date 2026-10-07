# Qwen bandwidth and attention diagnostics

Prepared 2026-10-07. These experiments do not change the qualified model.

The 44 CPU checks in `kernel-experiments-unit-v2.xml` pass. They cover work
partitioning, input circular-buffer capacities, attention page mapping and
per-user numerical checks. GDN hardware checks passed; attention v1 stopped on
an unsupported Python config alias before measuring the higher-precision modes.
The corrected v2 completed all 12 comparisons, but every mode still failed the
unchanged numerical limit. Both attempts are preserved in adjacent evidence
directories. HiFi4/FP32 reduces error but does not resolve it.

`qwen38-kernel-diagnostics-v1-20261007.service` runs persistently on the allocated
`.98` Galaxy; the run is now terminal. The exact launch and script are preserved
here. It waits at most
15 minutes for `qwen38-layer-profile-v5-20261007.service` to finish, then runs:

1. Attention: native math, HiFi4/FP32, and HiFi4/FP32 with accurate exponentiation
   at 8K/B1, 8K/B16, 128K/B8 and 262016/B4. Each user must meet the unchanged
   PCC >= 0.999 and relative-RMS <= 0.02 limits against FP32 attention on the
   same quantized KV inputs. A failed native mode does not prevent collecting
   the higher-precision comparisons.
2. GDN: one/two buffered input work items, one/two/four value partitions, and
   B1/8/16/32/64. Each variant also runs 4096 changing-input recurrences,
   64 near-identity decay steps, and an uneven 193-head test alternating two
   live allocations inside a trace. State remains FP32 and is read/written
   once per recurrence.

Each hardware command has a 35-minute outer timeout and uses the shared
device lock and recovery wrapper. The service has a 90-minute limit and
128-GiB host-memory cap. Attention failure is recorded without skipping the
independent GDN experiment. Neither result alone qualifies model accuracy.

Host receipts and logs are under `/home/ttuser/qwen38-artifacts-20261007/`:
`attention-precision-v1/`, `gdn-step-buffering-v1/` and the corresponding `.log`
files. The preserved script may be inspected or copied to a fresh result name;
rerunning it verbatim refuses to overwrite existing output directories.

## GDN results

All 30 geometry variants passed, with 96 long-horizon checkpoints, six decay
checks and six allocation/uneven-work trace checks. Four value partitions:

| Batch | One buffered item (us) | Two buffered items (us) | Latency reduction |
|---:|---:|---:|---:|
| 1 | 25.86 | 25.82 | 0.1% |
| 8 | 87.60 | 65.36 | 25.4% |
| 16 | 140.05 | 99.22 | 29.2% |
| 32 | 242.03 | 166.89 | 31.0% |
| 64 | 462.84 | 314.14 | 32.1% |

Every geometry still misses the P1 latency target. These are standalone
recurrence timings, including trace dispatch, with no full-model promotion.
The full receipt is compressed losslessly in `../gdn-step-buffering-v1/`.

## Profiler outcome

All six device cases passed. The generic Tracy importer subsequently hit the
128-GiB service memory cap while loading a 67.5-GB host-zone CSV. A CPU-only
recovery utility avoids this CSV and joins the existing native device timings
with cached op metadata. Its seven regression tests pass, but the real recovery
correctly rejects the exported metadata: only three of six model windows have
signposts. No stage timing is qualified from this capture. Originals are retained;
`../profile-recovery-v1/` records the outcome. Complete capture recovery or a
bounded recapture remains necessary.
