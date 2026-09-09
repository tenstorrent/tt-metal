# Why serving is slow: handoff

State of the investigation into Qwen3.8-27B serving performance on p300x2 (TP4).
Facts with how they were observed, then what is still unknown, then what I would
do next. Opinions are labelled.

The headline: **serving cost is device time, not software overhead, and decode is
~10x short of its target.** Within a decode step, 238.58 ms is device and 0.03 ms
is host. CI reproduces local numbers to within 2%, so nothing about the runner is
implicated.

## The numbers everything else refers to

| quantity | value | where measured |
| --- | --- | --- |
| decode step, device | **238.58 ms** | fence in `token_out_decode_step`, mean over 48 steps |
| decode step, host dispatch | **0.03 ms** | same fence (`exec_enq` + `sample_enq`) |
| sampler, device | **0.57 ms** | fence between the two traces |
| 48 GDN layers | ~113.7 ms (48%) | 2.3697 ms/layer x 48, `multichip_traced_decode.py --kind linear --batch 32` |
| 16 full-attention layers | ~10.0 ms (4%) | 0.6277 ms/layer x 16, same harness `--kind full` |
| **unattributed** | **~114 ms (48%)** | 238.58 - 123.7 - 0.57 |
| TPOT, CI | 244.41 ms | CI benchmark point, isl 2048 |
| TPOT, local | 238.6-244.1 ms | local vLLM runs |
| tput_user | 4.1 tok/s | 1000 / 244.41 |
| tput_user target | 41 tok/s | spec `perf_targets`; **10.0x short** |

## Facts

### F1. The decode step is device-bound

`QWEN36_DECODE_PROFILE=1` adds a `ttnn.synchronize_device` fence inside
`Qwen36Generator.token_out_decode_step` (in `tt/generator.py`, uncommitted at
handoff time -- see "Instrumentation" below).

```
STEP_PROF_MEAN over 48: model_device=238.01 sample_enq=0.04 sampler_device=0.57
```

Host enqueue is 0.03 ms of a 238.58 ms step. There is no dispatch overhead to
reclaim inside a model call.

### F2. Decode is correctly traced; it is not re-capturing per token

`QWEN36_DECODE_LOG_SETUP=1` counts `setup_token_out_decode` against decode steps:

```
total steps: 31   setups: 1
setup fired once:  reset_batch=True not_ready=True page_changed=True remap_present=True
steps 2..31:       page_changed=False remap_present=False
```

Trace capture happens once per batch, then `execute_trace` replays. An earlier
theory that `remap_changed = slot_remap is not None` (presence, not difference)
would re-trigger setup every step is **wrong** -- `slot_remap` is not passed on
steady-state steps.

### F3. The sampler is not the cost

0.57 ms of 238.58 ms, despite reducing `[32, 248320]` logits per token. An
earlier theory that on-device sampling over the large vocabulary dominated is
**wrong**.

### F4. CI matches local

CI benchmark point (isl 2048, osl 128, conc 1, 4 requests):

```
TPOT 244.41 ms   TTFT 40969 ms   E2EL 72009 ms   (= TTFT 41 s + 127 x 244 ms)
```

against local 238.6-244.1 ms. Within 2%. No runner-specific overhead exists.

### F5. Per-layer numbers reconcile to the step, so ~114 ms is unaccounted

`multichip_traced_decode.py` builds **one** `MultichipDecoder`, not the model
(verified: it calls `MultichipDecoder.from_state_dict`, and the file's harness
constructs a single decoder). So its 2.3697 / 0.6277 ms are per-layer:

```
48 x 2.3697 = 113.7 ms   (linear attention / GDN)
16 x 0.6277 =  10.0 ms   (full attention)
              123.7 ms   layers subtotal (52% of the step)
              114.2 ms   residual (48%)  <- UNIDENTIFIED
```

The residual covers embedding, 64 RMS norms, the final norm, any pad, and the
LM head.

### F6. The LM head slot loop does NOT fire in decode

`_project_lm_head_tile` loops per slot only when `hidden_states.shape[1] > 1`.
`MultichipModel.decode_forward` reshapes to `(1, 1, batch, hidden)` before
`terminal_forward`, so `shape[1] == 1` and a single 32-row projection runs. The
slot loop is a **prefill** path, and per-request prefill makes its batch 1 there
anyway.

A microbenchmark of the loop shape (32 sequential 32-row projections vs one)
measured **280.6 ms vs 8.5 ms, 32.9x** -- real, but not on the decode path. It
used plain `ttnn.linear`, not the model's DRAM-sharded program, so its absolute
values do not transfer.

### F7. Prefill cost per token is flat except at isl 2048

| isl | prefill | ms/token |
| --- | --- | --- |
| 128 | 1.14 s | 8.91 |
| **2048** | **41.0 s** | **20.00** |
| 65536 | 569.3 s | 8.69 |
| 131072 | 1150.4 s | 8.78 |

128, 65536 and 131072 agree closely. 2048 is 2.3x worse per token. Quadratic
attention would make *long* sequences worse, not a middle one, so this looks
like a bad shape or program selection near that length. **Unexplained.**

### F8. The agentic benchmark cannot finish in the job budget

From the CI log: `1/128 [22:03<46:42:16, 1323.91s/it]`, i.e. 128 items at
1323.91 s each = **47.1 h**, against `timeout-minutes: 1080` (18 h). The 128-item
loop is agentic trace replay (`llm_module/drivers/swo_bench_agentic_traces.py`),
not the configured performance point -- the only benchmark target for this model
is `isl=128 osl=128 conc=1 num_prompts=8`.

At the 41 tok/s target the same run projects to 4.8 h (if traces are 100% model
time) or 13.3 h (if 80%), i.e. inside budget.

### F9. The idle log lines are a logging artifact, not a fault

vLLM's stats logger (`loggers.py:310`) emits every ~10 s and picks the level by
activity: **active -> INFO, idle -> DEBUG**. Counting the successful evals run:

```
518 INFO + 49 DEBUG = 566 lines   -> 9% idle
gemma benchmarks (reference)      -> 11% idle
```

Both healthy. Local runs show no idle lines only because they are short and do
not surface DEBUG. The *proportion* is the metric, not the presence. The idle
fraction of the long agentic benchmarks job was never measured: GitHub returns
`BlobNotFound` for an in-progress job log, so it must be computed after the job
ends.

### F10. Idle stretches in the agentic run are client-side

The pasted excerpt shows `Running: 0 reqs, Waiting: 0 reqs` continuously from
17:02:35 to 17:06:05. Zero running *and* zero waiting means the client sent
nothing; if it were blocked on the model there would be a running request. So in
those windows the model is not the bottleneck. Whether that is agent scaffolding
(swo-bench replays recorded Claude-Code / Codex sessions, with tool calls between
turns) or a client timeout and backoff against our slow responses is
**unresolved** and needs the completed log.

## What is NOT the problem (each was tested and eliminated)

- host dispatch overhead (0.03 ms of 238.58 ms)
- decode trace re-capture (1 setup / 31 steps)
- eager/untraced decode (replay path confirmed)
- on-device sampling (0.57 ms)
- the LM head slot loop, on the decode path (F6)
- CI/runner overhead (F4)
- page-table churn or slot remap churn in steady state (F2)

## Open questions, in the order I would take them

1. **The ~114 ms residual (48% of decode).** Biggest single unknown. Attribute it
   per-op with tracy on a *standalone* full-model traced decode at batch 32 --
   no server needed, so no vLLM in the way. If it is not the LM head (F6 says it
   should not be), the candidates are the 64 RMS norms, the embedding, and the
   pad in `terminal_forward`.
2. **The isl 2048 outlier (F7).** 2.3x worse per token than its neighbours.
   Cheap to reproduce standalone with `kactive.py --length 2048`, then sweep
   nearby lengths to find the boundary.
3. **The GDN layers at 2.3697 ms each.** 48 of them, 48% of decode. This is the
   dominant *known* cost. Whether 2.37 ms is reasonable for one GDN layer at
   batch 32 on TP4 has never been assessed against a roofline.
4. **The client-side idle in the agentic run (F10).** Needs the completed job log:
   compute the idle fraction as in F9, and grep the swo-bench output for
   timeouts and retries.

## Instrumentation available

All env-gated and off by default. Two are uncommitted at handoff -- check
`git status` and commit or discard deliberately.

| env var | what it does | where |
| --- | --- | --- |
| `QWEN36_DECODE_PROFILE=1` | per-step fences: model device time, sampler enqueue, sampler device time | `tt/generator.py::token_out_decode_step` (uncommitted) |
| `QWEN36_DECODE_LOG_SETUP=1` | logs setup-vs-step counts and which condition fired | `tt/generator_vllm.py::decode_forward` (uncommitted) |
| `QWEN36_PREFILL_LOG_K=1` | logs active rows per prefill call | `tt/generator.py::prefill_forward` (committed) |
| `QWEN36_PREFILL_PER_REQUEST` | one request per prefill call; **1 in the CI spec** | `tt/generator.py::prefill_forward` (committed) |
| `QWEN36_PREFILL_SCAN` | `sequential` (default) or `hillis` | `tt/functional_decoder.py` (committed) |
| `QWEN36_SCAN_MATMUL_GRID` | scan matmul core grid, `0` disables the tuned program | `tt/functional_decoder.py` (committed) |

Scratch harnesses used (paths are session-scoped; copy them somewhere durable):
`kactive.py` (standalone prefill at k active rows, `--scatter`, `--device-logits`,
`--all-layers`, `--max-context`), `equiv.py` (batched vs per-request prefill
equivalence), `lmhead.py` (LM-head slot-loop microbenchmark).

## Method notes that cost time

- **A `SIGKILL` of a device run corrupts device 3** (`Read 0xffffffff over PCIe ID
  3: the board should be reset`). Always `tt-smi -r` and settle ~2 minutes after
  any kill, and health-gate before trusting a measurement -- an MMIO-timeout run
  silently produced a bogus result once.
- **Do not run two device things at once.** One test was invalidated by a
  `killruns` issued for another.
- **In-progress CI logs are not fetchable** (`BlobNotFound`); wait for completion.
- **`multichip_traced_decode.py` measures one layer, not the model.** Comparing
  its number against a whole-model step produced a bogus "389x overhead" claim
  that survived several rounds of reasoning before being caught. Check what a
  harness builds before using its number as a baseline.
