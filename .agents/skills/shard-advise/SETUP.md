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
(operator setup; do not build tt-mlir from inside a model experiment).

**Required tt-mlir branch:** `mvasiljevic/shard-advisor-dram-sharding` on
`github.com/tenstorrent/tt-mlir`, pinned at commit `618cd4e75d`. It is
`ttnn-jit-shard-advisor` (validated at `dcb25113` or later) plus the DRAM-sharded-matmul
optimizer integration, without which DS advice is unreachable for ttnn-traced decoders and
the advisor's only matmul lever is 1D-mcast. It also fixes `kNumDRAMBanks`, which was
hardcoded to 12 (Wormhole); QB2 is Blackhole with 8, and at 12 the DS weight tensor cannot be
allocated at all. Pin this exact commit so runs are comparable. This branch carries the `ttnn-advise` CLI and the
ttnn-jit interception tracer with the decode-op handlers this skill relies on (paged cache +
SDPA-decode, qkv split/concat, rope, etc. — see A.3); `main` does not have them. If the advisor
blocks on an op it doesn't model yet, add a tracer handler per A.3 (and ideally upstream it to this
branch).
```
git clone -b mvasiljevic/shard-advisor-dram-sharding https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir
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
export TTMLIR_ADVISOR_HOME=/path/to/tt-mlir     # MUST be exported BEFORE sourcing; bootstrap.sh
                                                # checks it first and exits 1 if unset
cd "$TTMLIR_ADVISOR_HOME"
source tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh \
  >/dev/null 2>&1                               # redirect, never | tail
python3 -c "import ttnn; print(ttnn.__file__)"  # must resolve, not error
```
`bootstrap.sh` activates the env, sets `SYSTEM_DESC_PATH`, and runs `ttrt query --save-artifacts`
once to make the system descriptor.

If it reports `ttnn-advise` missing, **look for the binary before concluding anything**:
`ls $TTMLIR_ADVISOR_HOME/build/bin/ttnn-advise`. That is where a completed A.1 puts it, and it is
frequently present while absent from `PATH`. This inference has been made wrongly twice in this project,
in both directions — once concluding the advisor was unbuilt, once concluding the binary did not exist
anywhere — and both times from reasoning about the tool instead of listing the build tree.

> **The path above is the tt-mlir one, and it matters.** Inside a **tt-metal** checkout this skill
> lives at `.agents/skills/shard-advise/scripts/bootstrap.sh`, and earlier revisions of this file gave
> that path here — but `$TTMLIR_ADVISOR_HOME` is a **tt-mlir** checkout, where the script ships under
> `tools/ttnn-jit/integrations/agentic-research/`. Sourcing the tt-metal path from tt-mlir gives
> `No such file or directory`, and the follow-up import then gives
> `ModuleNotFoundError: No module named 'ttnn'`. Together those look exactly like a broken or missing
> advisor install; they are not. This cost one operator a wrong "the advisor is unusable" conclusion
> and a day of the advise arms being held for no reason. If you see that pair, check this path and
> `TTMLIR_ADVISOR_HOME` before concluding anything about the build.

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

**B.1 Point a capture target at the decoder.**

> **For stage 02b (`$advisor-challenger`), copy that stage's
> `.agents/skills/advisor-challenger/scripts/capture_template.py` instead of `advise_decoder.py`.**
> `advise_decoder.py` is written for one model, and the reuse list below omits the **precision policy** —
> three capture scripts written from it constructed the decoder with no dtype argument and therefore traced
> the CLASS DEFAULTS rather than the precision the model ships. On one model that traced bf16 attention for
> a cell shipping bfp8, excluding two matmuls from DRAM-sharding consideration for a dtype never used, and
> the missing win was worth -10%. The template exists to close exactly this hole and records what it traced
> so a gate can verify it.

Copy `scripts/advise_decoder.py` to `advise_<model>.py` and edit it to build one decode step of this
model's `OptimizedDecoder`, **constructed with the SHIPPED precision policy** — sourced from what executed,
never `resolved_policy.constructor_defaults` — reusing the model's own `tests/test_optimized_decoder` input
builders (config, synthetic state dict, paged KV cache, rope, `current_pos`). A synthetic state dict is
fine: the advisor reasons about layout, not values. What must be real is the **dtype** and the **shapes**.
Expose `make_inputs(device)` and `decode(hidden)`.
**Append** the snapshot root + tt-metal to `sys.path` (never prepend — tt-metal's `ttnn/` dir
shadows the real package). Pick a representative dense layer (attention + dense MLP); one target
per distinct layer kind.

**B.2 Run the advisor:**
```
export PYTHONPATH=<snapshot-root>:<tt-metal>:$PYTHONPATH
cd "$TTMLIR_ADVISOR_HOME"
ttnn-advise capture advise_<model>.py:decode --out <out-dir> \
  --pipeline-options allow-bf16-dram-sharded-matmul=true
```
**Pass `allow-bf16-dram-sharded-matmul=true` whenever any traced weight is bf16.** Without it bf16 weights
are declined *by policy*, not by capability — bf16 DS runs at PCC 1.0000 — and one cell got 0-of-5 DS
matmuls advised for precisely this reason.

**Write `--out` into the cell tree, not `/tmp`**, so the advice is an artifact of the run and not a
scratch file: stage 02b requires `doc/advisor_challenger/shard_advise/<layer_kind>/`, which is where its
gate looks. **Do not redirect stderr to `/dev/null`** — a capture that blocks on an unhandled op then looks
like a success.

What is where, corrected: `report.json` carries `ops[]` (index, op, layout including `cores=`, and the
`program_config` **family name**) and the full `reshards[]` list with each edge's producer, consumer and
from/to layouts — that is enough to reconcile advice against a profile, and 02b's `reconcile.py` reads
only this file. `final_ir.mlir` is authoritative for what `report.json` does **not** carry: the matmul
program-config **parameters** (grid, `in0_block_w`, `per_core_N`, `out_subblock_w`), the required matmul
input layout, and any layout question the reconciliation flags as unresolved. If capture blocks on an op,
do A.3, re-sync, re-run.

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

**B.4 Measure & decide.**

> **Stage 02b overrides B.3 and B.4.** It is a contribution measurement against a *frozen* incumbent, so
> its ship rule is non-overlap of n=5 timed repeats against a recorded noise floor, not "beats the best
> measured candidate", and it derives its own geometry rather than copying advised configs. Follow
> `advisor-challenger/SKILL.md` there. B.3's adaptations below remain accurate engineering constraints and
> are worth reading either way.

PCC baseline vs advised (open with
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
