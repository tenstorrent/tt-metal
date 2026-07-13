# Shard-Advise — setup & use (incorporate into an optimize pass)

The advisor is a build/optimization-time L1-layout planner (`ttnn-advise` CLI), not a runtime
hook — no model test invokes it. An agent runs it **during the `optimize` / `03-optimized`
goal** (see `optimize/SKILL.md` OPT-015), reads its recommendation, hand-applies it into
`optimized_decoder.py` as a **candidate**, then measures with `tt-perf-report` and keeps it only
if it wins. This doc is the reproducible recipe: Part A is one-time per machine, Part B is per
model each optimize pass, Part C is how to fold it into the loop. Everything here was exercised
across ~40 decoders; a fresh agent can reproduce it. The advisor is not polished — expect a
bounded tracer-handler fix now and then (Part A.3) — that's normal, not a blocker.

---
## Part A — one-time, per machine (before first use)

**A.1 Build the advisor env.** A tt-mlir checkout built with the OpModel + ttnn-jit stack
(operator setup; do not build tt-mlir from inside a model experiment):
```
cmake -G Ninja -B build -DTTMLIR_ENABLE_OPMODEL=ON -DTTMLIR_ENABLE_TTNN_JIT=ON \
  -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
cmake --build build            # installs ttnn-advise + ttnn_jit into the toolchain venv
export TTMLIR_ADVISOR_HOME=/path/to/tt-mlir
```
Its venv `ttnn` must resolve to `tt-mlir/third_party/tt-metal/src/tt-metal` (a symlink to your
tt-metal) so the advisor traces the same ttnn your model runs. See tt-mlir getting-started for
full prerequisites.

**A.2 Verify activation works** (also the per-shell step, B.0):
```
cd "$TTMLIR_ADVISOR_HOME"
source .agents/skills/shard-advise/scripts/bootstrap.sh >/dev/null 2>&1   # redirect, never | tail
python -c "import ttnn; print(ttnn.__file__)"    # must resolve, not error
```
`bootstrap.sh` activates the env, sets `SYSTEM_DESC_PATH`, and runs `ttrt query --save-artifacts`
once to make the system descriptor. If it reports `ttnn-advise` missing, A.1 wasn't done.

**A.3 Tracer handlers (know this exists).** The advisor builds TTIR by monkeypatching ttnn ops
in `tools/ttnn-jit/_src/interception_tracer.py`. A model may use an op it doesn't model yet →
capture blocks on that op. Fix is bounded per-op: add a handler emitting the matching TTIR op
(shape/dtype only), then sync:
```
cp "$TTMLIR_ADVISOR_HOME/tools/ttnn-jit/_src/interception_tracer.py" \
   /opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/ttnn_jit/_src/interception_tracer.py
```
Handlers already cover the common decode ops (linear, rms_norm, reshape, slice/`__getitem__`,
transpose/permute — emit `ttir.permute`, not `ttir.transpose`; softmax, topk, where, scatter,
zeros_like, broadcast binaries, negative-dim reductions, unsqueeze_to_4D, qkv split/concat,
paged cache + SDPA-decode). **Terminal (no TTIR op → skip these paths):** `ttnn.sparse_matmul`
(batched-MoE experts) and SSM/gated-delta ops (`softplus`, `prefix_scan`, `hc_sum_reduce`,
`assign`). Best long-term fix is upstreaming the handlers into tt-mlir so this step disappears.

---
## Part B — per model, each optimize pass

**B.0** Activate the advisor env (A.2) in the shell.

**B.1 Point a capture target at the decoder.** Copy `scripts/advise_decoder.py` to
`advise_<model>.py` and edit it to build one decode step of this model's `OptimizedDecoder`,
reusing the model's own `tests/test_optimized_decoder` input builders (config, synthetic state
dict, paged KV cache, rope, `current_pos`). Expose `make_inputs(device)` and `decode(hidden)`.
**Append** the snapshot root + tt-metal to `sys.path` (never prepend — tt-metal's `ttnn/` dir
shadows the real package). Pick a representative dense layer (attention + dense MLP); one target
per distinct layer kind.

**B.2 Run the advisor:**
```
export PYTHONPATH=<snapshot-root>:<tt-metal>:$PYTHONPATH
cd "$TTMLIR_ADVISOR_HOME"
ttnn-advise capture advise_<model>.py:decode --out /tmp/<model>-advice 2>/dev/null
```
Read `/tmp/<model>-advice/final_ir.mlir` (**authoritative**). `report.json`/stdout are summaries;
the per-op `program_config`, the required matmul **input layout**, and the advisor's reshards
are only in `final_ir.mlir`. If capture blocks on an op, do A.3, re-sync, re-run.

**B.3 Apply as a candidate** into `optimized_decoder.py` (or a sibling variant for a clean A/B).
For each `ttnn.linear`, take the `matmul_multi_core_reuse_multi_cast_1d` config from `final_ir`
(grid, `in0_block_w`, `per_core_N`, `out_subblock_w`) + width-sharded L1 output, with these
required adaptations (learned; honor them or it won't run / regresses):
- feed the matmul **input L1-interleaved** (width-sharded input to `mcast_in0` fails);
- **clamp `out_subblock_w`** to the decoder's compute-kernel register budget (advisor assumes 8;
  `fp32_dest_acc_en=True` caps 4, else `available_reg_count` fatal);
- if baseline weights are **DRAM-sharded**, make DRAM-interleaved weight copies for this path;
- **replicate the advisor's own `to_memory_config` reverts** at head-split / SDPA / residual
  boundaries (they're in `final_ir`);
- optionally apply the advised width-sharded `rms_norm` / residual-add / `concat_heads` layouts
  as one chain — measure it; it's often ~neutral.

**B.4 Measure & decide.** PCC baseline vs advised (open with
`ttnn.open_mesh_device(MeshShape(1,1))` = full 8x8 grid the advisor assumes), then traced-decode
`tt-perf-report` before/after. Keep the advised config only where it beats the DRAM-sharded /
best measured candidate (OPT-004). Note: `tt-perf-report` splits `MatmulDeviceOperation` into
`(in0:l1_interleaved)` + `(in0:width_sharded)` rows — **sum** them.

---
## Part C — incorporate into the optimize loop (now)

During the `03-optimized` goal, after `$graph-fusing` and the operation-topology audit, on the
dense attention+MLP block: do B.1-B.2 to get `final_ir`, seed it via B.3 as the first candidate,
then let the normal optimize search (OPT-003 residual chain, OPT-004 DRAM-sharded sweep,
precision) iterate on top and keep the measured winner. Re-query the advisor only if you rewrite
the block. It's a **seed, dense-path only, one candidate** — never a replacement for the
DRAM-sharded matmul search. That is the whole incorporation: no runtime hook, no new stage.
