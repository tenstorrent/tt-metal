# Getting Started with tt_hw_planner

New here? Start with this. For every internal stage, flag, and design detail, see **`README.md`** next to this file.

## What this tool does
You have an AI model on HuggingFace. You want it to **run on Tenstorrent hardware, correctly and fast**. Normally an engineer would rewrite the whole model by hand for the chip — weeks of work. This tool does that automatically: it rewrites the model piece by piece, checks each piece gives the same answers as the original, and then tunes it for speed. You mostly just start it and wait.

## The short version
If you read nothing else:

1. **Set up once** — build `ttnn`, install the agent dependencies, log into Claude. *(→ Before you start)*
2. **Bring it up** — `auto-up <model> --box <B> --mesh <M>` rewrites the model into native TTNN and verifies every piece on the device; `promote` resumes any leftovers, then `emit-e2e` wires the graduated pieces into the full end-to-end pipeline. *(→ Section 1)*
3. **Make it fast** — `optimize <model> --devices all` tunes the whole pipeline toward the hardware limit (or `optimize <model> --module-level` tunes each graduated module first), keeping only verified speedups. *(→ Section 2)*

That's the whole path. The rest of this guide is detail and troubleshooting. The diagram below shows the same flow, in full.

## The flow at a glance

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#111111','primaryBorderColor':'#94a3b8','lineColor':'#475569','fontFamily':'Segoe UI, Arial, sans-serif','fontSize':'13px'}}}%%
flowchart TD
  subgraph BG[" "]
  direction TB
    IN(["auto-up &lt;model&gt; · --box T3K · --mesh 2x2 · --reverify (rerun-safe)"]) --> REG["registry sync + drift check — remote-first · non-fatal"]
    REG --> S1["STEP 1 · plan + compat — REUSE / ADAPT / NEW split"]
    S1 --> S2["STEP 2 · scaffold — clone closest demo family · inline auto-onboard if arch unknown"]
    S2 --> S3["STEP 3 · LLM gate — does bring-up need an agent?"]
    S3 --> S4["STEP 4 · phase-1 autofill — CPU torch fallbacks + aliases"]
    S4 --> S5["STEP 5 · prepare — build the runnable pytest invocation"]
    S5 --> S6["STEP 6 · bring-up cc engine — single agent driven by the bringup_mcp gate"]
    S6 --> CAP["capture real inputs — per-component HF I/O"]
    CAP --> OVL1["load overlays — replay prior bring-up"]
    OVL1 --> PRE["pre-flight PCC · ttnn import check"]
    PRE --> RECON

    subgraph ITER["BRING-UP cc engine · per-component gate · single agent · PCC ≥ 0.99 on device"]
    direction TB
      RECON["loop-start reconciliation<br/>restore stale tests · reinject + recompose ready parents"] --> PICK["gate picks TARGET<br/>termination_check names {component · rung}"]
      PICK --> AG["🤖 single claude agent · isolated worktree<br/>works next_target.rung — emit/repair · fix_harness · resolve_loader · shard"]
      AG --> APP["record_result + ttnn-import<br/>native-only · torch-delegating stub refused"]
      APP --> PCC{"PCC ≥ 0.99<br/>on device?"}
      PCC -->|pass| GRAD["graduate ✓ · .last_good_native<br/>(shard → .last_good_sharded)"]
      GRAD --> OVL2["overlays index + patch"]
      OVL2 --> SW["regression + validation sweep<br/>compute split X / N on device"]
      PCC -->|fail| RB["restore_best / rollback<br/>per-component attempt budget"]
      RB --> WJ{"budget left?<br/>(brain G8 · extend-cap)"}
      WJ -->|retry| PICK
      WJ -->|exhausted| DEC["decompose parent → children ·<br/>fallback-to-CPU · mark_manual + lock"]
      DEC --> PICK
    end
    SW --> MORE{"more components<br/>left?"}
    MORE -->|"yes · budget left"| RECON
    MORE -->|"leftovers · iter budget capped"| PROM["promote — resume cc bring-up of the<br/>REMAINING components (fresh session · replays overlays)"]
    PROM --> RECON
    MORE -->|"all graduated"| FC["final categorization — ON_DEVICE / CPU_REUSE /<br/>KERNEL_MISSING / PENDING"]

    FC -. "optional · separate" .-> OVL3["overlays — apply (replay) · or dispose / drop"]
    FC -. "optional · per-module" .-> MOPT["🤖 optimize --module-level — tune graduated modules one at a time<br/>module-scoped perf test · --then-e2e confirms in the pipeline"]

    subgraph E2E["EMIT-E2E — build the end-to-end pipeline"]
    direction TB
      BLD["🤖 BUILDER wires graduated stubs → pipeline<br/>+ exposes host_op_selftest()"] --> GR{"e2e gate · termination_check<br/>G1–G6 + host-op observer"}
      GR -->|FAIL| FIX["🤖 FIXER closes holes"]
      FIX --> GR
    end
    FC --> BLD
    GR -->|PASS| PROF

    subgraph OPT["OPTIMIZE · reuse-first · deterministic gate + kernel ladder · learns across runs · every pipeline"]
    direction TB
      PROF["discover pipeline(s) from demos + auto-generate missing perf test<br/>profile on device (tracy)"] --> OTC{"termination_check<br/>next_target = {op, op_class, rung}"}
      OTC -->|"blocking op · gate names next_target.rung"| RECALL["🤖 recall_knobs(op_class, grid, bound)<br/>read THIS op's catalog slice FIRST — tuned levers ranked first"]
      RECALL --> L1["🤖 rung 1 · knob:grid<br/>occupy the full core grid"]
      L1 -->|"else"| L2["🤖 rung 2 · knob:dtype<br/>weights bf16 → bf8_b → bf4_b"]
      L2 -->|"else"| L3["🤖 rung 3 · tt-lang kernel<br/>custom kernel via Python DSL"]
      L3 -->|"else"| L4["🤖 rung 4 · C++ Metalium kernel<br/>raw generic_op kernel"]
      L4 -->|"else · matmul still memory-bound on a mesh"| L5["🤖 rung 5 · tp-fracture · tensor-parallel<br/>tp_pick_degree → shard weight across chips + CCL → verify_tp_fracture"]
      L5 -->|"else"| L6["🤖 rung 6 · structural<br/>investigate arch for reducible work"]
      L6 --> OG{"PCC ok ·<br/>faster · honest?"}
      OG -->|no| RV["revert / discard"]
      OG -->|yes| CMT["COMMIT speedup ✓ — scoped git"]
      CMT --> DIST["🤖 distill_knob — write the IMPROVISED win back<br/>+ graduate a reused provisional → trusted"]
      RV --> REC["record_kernel_attempt (win or lose)"]
      DIST --> REC["record_kernel_attempt (win or lose)"]
      REC --> PROF
      CAT[("knob catalog<br/>GUIDELINES + LEARNED_* / GRADUATED_*")] -. "reuse known knob" .-> RECALL
      DIST -. "provisional · graduates on a different model" .-> CAT
    end
    MOPT -. "reuses the ladder per module" .-> OTC
    OTC -->|"can_stop · every op at floor or ladder exhausted"| DONE(["model runs natively on device — correct AND fast"])
  end

  classDef agent fill:#fde68a,stroke:#b45309,color:#111;
  classDef gate  fill:#fef3c7,stroke:#a16207,color:#111;
  classDef ovl   fill:#dcfce7,stroke:#15803d,color:#111;
  classDef good  fill:#bbf7d0,stroke:#15803d,color:#111;
  classDef warn  fill:#fecaca,stroke:#b91c1c,color:#111;
  classDef term  fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef build fill:#e0e7ff,stroke:#6366f1,color:#111;
  classDef prep  fill:#cffafe,stroke:#0891b2,color:#111;
  classDef iter  fill:#ede9fe,stroke:#7c3aed,color:#111;
  classDef post  fill:#fae8ff,stroke:#a21caf,color:#111;
  classDef opt   fill:#ffedd5,stroke:#ea580c,color:#111;
  class AG,BLD,FIX,PROM,MOPT,L1,L2,L3,L4,L5,L6,RECALL,DIST agent;
  class PCC,WJ,MORE,GR,OG,OTC gate;
  class OVL1,OVL2,OVL3,CAT ovl;
  class GRAD,CMT good;
  class RB,DEC,RV warn;
  class IN,DONE term;
  class S1,S2,S3,S4,S5,S6 build;
  class CAP,PRE,REG prep;
  class RECON,PICK,APP,SW iter;
  class FC post;
  class PROF,REC opt;
  style BG fill:#ffffff,stroke:#ffffff;
  style ITER fill:#f1f5f9,stroke:#cbd5e1,color:#334155;
  style E2E fill:#f1f5f9,stroke:#cbd5e1,color:#334155;
  style OPT fill:#f1f5f9,stroke:#cbd5e1,color:#334155;
```

Each speed lever is only tried when the cheaper one above it is used up, and every change is kept **only** if it's faster *and* still gives the right answers. You don't pick these — the tool does. (TP, step 5, only kicks in for a big matrix-multiply that's still slow after the earlier steps.)

## Optional: drive it from the dashboard (web UI)

Prefer not to SSH into machines and babysit `tmux`? A companion web dashboard —
**"Bring Up"** — runs everything in this guide from a browser, across several
machines at once. It just issues the same `python -m scripts.tt_hw_planner …`
commands remotely, so this README stays the source of truth for stages/flags.

**What it does**
- **Provision a machine from scratch** — clone this fork + `feature/tt-hw-planner`,
  `git submodule update`, `build_metal.sh`, `create_venv.sh --skip-compat-check`,
  install `tt-lang`, then verify `tt_hw_planner` — with a live build log.
- **Bring up → optimize** — pick a model (or search HuggingFace by name), a machine,
  `--box`/`--mesh`, choose stages (`auto-up` → `emit-e2e` → `optimize`) and flags,
  and launch. Each run streams its log in its own card.
- **Many models at once** — launch different models on different devices
  concurrently; each run has its own live log, **RUN_REPORT.md** viewer, and an
  embedded live **optimize dashboard**.
- **Sync the tool** — one-click `git fetch` + `pull --ff-only` of this repo on a
  target machine.
- Optional **Telegram** control + notifications.

**Run it**
```bash
git clone git@github.com:apande-TT/cc-dashboard.git
cd cc-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8001
```
Open <http://127.0.0.1:8001>. It reads the machines in your `~/.ssh/config`; use
each machine's **Connect** button once (installs your SSH key) so it can drive
that machine. See the dashboard repo's README for `.env`/Claude-auth setup.


## Before you start — prerequisites
Do these once, in order, in a **fresh standalone clone** (not a linked git worktree). Following the commands literally works — the known traps are baked in. If something is still missing, the tool stops and prints the exact fix.

### Machine prerequisites (before cloning)
- A Tenstorrent board is visible: `ls /dev/tenstorrent` shows device nodes.
- SSH access to GitHub works.
- The **`claude` CLI is installed and logged in** — `cc` uses native Anthropic auth, so there is **no `.env.agent`**:
  ```bash
  curl -fsSL https://claude.ai/install.sh | bash   # only if not already installed
  claude                                            # log in once
  ```
  (Or instead `export ANTHROPIC_API_KEY=…`.)

### Build the environment (run in order)

**1. Clone from your home dir** — NOT inside an existing repo:
```bash
cd ~ && git clone git@github.com:apande-TT/tt-metal.git tt-metal-xtts && cd tt-metal-xtts
```

**2–3. Check out the tool branch, then make your model branch from it:**
```bash
git checkout feature/tt-hw-planner
git checkout -b xtts-v2-bringup
```

**4. Init submodules** — required for the build (the clone does NOT do this):
```bash
git submodule update --init --recursive
```

**5. Create the venv — `--skip-compat-check` is REQUIRED.** A harmless dependency version clash otherwise makes the script exit 1 even though the venv is fine:
```bash
./create_venv.sh --skip-compat-check
```

**6. Build tt-metal and verify** (compiles the C++ side):
```bash
./build_metal.sh
source python_env/bin/activate
python -c "import ttnn; print('ttnn ok')"     # must print: ttnn ok
```
> **Re-run this every time you update the branch** (`git pull` / `git merge`). Pulling new code changes the *source* but does **not** recompile it — running on a stale build causes silent wrong answers (tests quietly read as "OTHER").

**7. Install `tt-smi` in its OWN venv** — **never** into the tt-metal venv (it drags in an older `tt-umd` that corrupts the device layer). The tool auto-discovers `~/.tenstorrent-venv/bin/tt-smi`:
```bash
python3 -m venv ~/.tenstorrent-venv
~/.tenstorrent-venv/bin/pip install -U pip tt-smi
export PATH="$HOME/.tenstorrent-venv/bin:$PATH"   # add this line to your ~/.bashrc too
tt-smi -s | head                                   # must show your board
```

**8. Confirm transformers matches tt-metal's pin.** The version is **not** hardcoded — the tool reads it from `tt_metal/python_env/requirements-dev.txt`, so it tracks upstream bumps (`5.12.1` at the time of writing). `setup_env.sh` (step 10) checks this and prints the exact `uv pip install` fix; if bring-up offers to downgrade transformers, decline it or pass `--no-env-fix`:
```bash
python -c "import transformers; print(transformers.__version__)"   # must match requirements-dev.txt
```

**9. Clear any stale kernel cache:**
```bash
rm -rf ~/.cache/tt-metal-cache "$TT_METAL_HOME/.cache/tt-metal-cache"
```

**10. Verify the whole environment in one shot.** You must **`source`** it (do not execute it — it needs to set env vars in your shell):
```bash
source models/experimental/perf_automation/setup_env.sh    # must print: environment ready
```
It self-detects the checkout, installs any missing agent deps, and checks ttnn/torch, the `claude` CLI + auth, `tt-smi`, transformers against tt-metal's pin (read live from `requirements-dev.txt`), and tt-lang. If any line says **FAIL**, fix it before continuing. Safe to re-run anytime.

**11. Now run the tool** — bring-up → emit-e2e → optimize, all on this branch (→ Section 1 & 2 below).

### tt-lang kernel rung — know this before `optimize`
`tt-lang` (`ttl`, the optimize **rung-3** custom-kernel lever) ships **`cp312` wheels only**, but this venv is **Python 3.10** (`create_venv.sh` default) — so it cannot install, and the tool silently skips that rung.
- **Bring-up does not need tt-lang at all**, and optimize still has the grid / dtype / C++ rungs. Staying on 3.10 is fine.
- Only if you specifically want tt-lang kernels during optimize: rebuild the venv on **Python 3.12** — `create_venv.sh` supports it (`./create_venv.sh --python-version 3.12`).

### Extra prerequisites before `optimize` (not needed for bring-up)
- You are in a **standalone clone** (this is one) — never run `optimize` from a linked git worktree (kernel JIT mixes worktree `.cpp` with main-tree `.hpp` → no trace).
- **Commit the model dir to git first** — optimize's REVERT needs a clean baseline.
- Pass the right `--devices` for your board (e.g. `0,1`); a wrong partial spec trips a fabric error.
- Perf is measured **trace + 1 command queue** end to end — the tool opens the device with a single CQ; there is **no 2-CQ track**.
- `TT_PERF_TRACE` selects the run mode: defaults to `1` (trace mode on) when unset; set `0` for eager mode.

### Handled automatically (don't chase these)
profiler orphan-marker heal + `libtt_metal.so` rebuild, marker-buffer drain, CSV extraction, device reset (`tt-smi -r`), crash/hang/stale-CSV guards, git checkpoint/revert, fabric-wedge avoidance, the tt-lang auto-install attempt, and CPU stubs for GPU-only packages (`flash_attn`, `mamba_ssm`, …).

Run everything from the `tt-metal-xtts/` folder.

## Section 1 · Bring up any model

Get a model running correctly on the chip. Three commands, run in order.

> Optional first look (changes nothing): `python -m scripts.tt_hw_planner plan <org>/<model>` — prints the memory-fit verdict and what's already supported vs. needs porting.

**Step 1 · `auto-up` — the one-command bring-up (always start here)**
```bash
python -m scripts.tt_hw_planner auto-up <org>/<model> --box QB2 --mesh 2,2
```
Plans, scaffolds a demo, captures real inputs, then runs the **cc engine** — a single Claude agent driven by the per-component `bringup_mcp` gate that ports each component to native TTNN, PCC-testing every piece on the device and graduating the ones that pass. **When to use:** the first step for any new model. Only `--box` and `--mesh` are required; it locks in the agent and iteration budget for you. It runs a while — use `tmux`/`nohup`.
- `--box` = one of `N150 N300 T3K QB2 Galaxy GalaxyBH`
- `--mesh` = the **physical hardware chip arrangement**, e.g. `2,2` (4 chips in a square) or `1,4` (4 in a row); must be a *canonical* shape for `--box`, and `2,2` and `2x2` are equivalent. (This is the literal device mesh — different from optimize's `--mesh`, which is a TP×DP topology; see Section 2.)
- `--reverify` = re-run every component's PCC gate from scratch (clears the graduation markers but keeps the ported code), so a rerun on an updated build re-earns graduation honestly instead of trusting a stale marker.

**Step 2 · `promote` — resume if bring-up didn't finish**
```bash
python -m scripts.tt_hw_planner promote <org>/<model> --box QB2 --mesh 2,2
```
`auto-up` caps at 24 iterations. If some components didn't graduate, `promote` re-runs the loop **only on the leftovers** (already-graduated components keep their snapshots and aren't re-attempted). **When to use:** after `auto-up` if not everything graduated — run it repeatedly, progress accumulates each pass. Same required `--box`/`--mesh`.

**Step 3 · `emit-e2e` — wire the pieces into the full pipeline**
```bash
python -m scripts.tt_hw_planner emit-e2e <org>/<model>
```
Once all components are graduated, a BUILDER agent wires them into the end-to-end task pipeline (exposing a `host_op_selftest()` hook), then a deterministic **e2e gate** — `termination_check` over gates G1–G6 (including the G5 host-op observer that proves the forward runs fully on device, and the G6 trace gate) — drives a FIXER loop until it passes on device. **When to use:** after everything is graduated, to produce a working end-to-end model. Handy flags: `--task <t>` / `--all-tasks` (multi-task models), `--max-grade-rounds`, `--pcc-target`.

### Overlays — save & replay a model's graduated work
An **overlay** is the captured set of file changes a bring-up produced — the per-component `_stubs/` (the graduated native-TTNN code) plus any patches. When a run is worktree-isolated the tool **auto-captures** them, so a model can be **replayed later without re-running the LLM**.
```bash
python -m scripts.tt_hw_planner overlay-list   <org>/<model>   # what's stored (omit model = all)
python -m scripts.tt_hw_planner overlay-apply  <org>/<model>   # replay the graduated modules onto a clean tree
python -m scripts.tt_hw_planner overlay-revert <org>/<model>   # undo an apply
python -m scripts.tt_hw_planner overlay-drop   <org>/<model>   # discard the stored overlays (omit model = wipe all)
```
Use **apply** to restore a previously brought-up model's graduated modules; **drop** only loses the replay shortcut, never the tool itself.

## Section 2 · Optimize

Once a model runs correctly, `optimize` profiles it on the device and climbs a per-op speed ladder — **cheap knobs (`grid` / `dtype` / `shard` i.e. tensor-parallel / `fidelity`) → algorithmic/structural restructure (the `host` lever — trace · fusion · gather · KV-cache — deliberately tried *before* any hand-written kernel) → tt-lang kernel → C++ Metalium kernel** — committing **only** PCC-verified, genuinely faster changes.

The **knob** rung is not a fixed sequence. The tool classifies each op as **memory-**, **compute-**, or **dispatch-bound** (or **unknown/other** when the roofline can't tell), and uses that classification only to set the *priority* — which knob is tried first — **never to drop a knob**. Every applicable knob is still tried at least once: after the bound-appropriate knobs are exhausted, a completeness sweep offers each remaining one before the op may be declared done (a bound estimate is only a hint, and ops are rarely purely one-bound). Priority order per bound:
- **memory-bound** (and **unknown/other**, the default): `grid → dtype → shard → fidelity`
- **compute-bound** and **dispatch-bound**: `grid → fidelity → dtype → shard`

(`grid` applies until the core grid is full; `dtype` only to matmuls; `shard` and `fidelity` always apply. `tp-fracture` — sharding a still-memory-bound matmul across the mesh — is offered after the knobs for eligible ops.)

### How to read the optimize report (`RUN_REPORT.md`)
Each optimize run writes a per-op ladder table into `RUN_REPORT.md` in the model dir. One row per op, one column per lever, and a final `best ms` (the op's best measured device time so far):
```
op                        grid   fidelity  dtype   shard   host   tt-lang   cpp    other    best ms
Matmul 128x14336x4096     ·try   —         ✓win    ✓win    ·try   ·try      ·try   ·try     1061.00
```
Columns (this is the full lever set — the same ladder, one column each):
- **`grid`** — core-grid occupancy (spreading work across more cores).
- **`fidelity`** — math fidelity (HiFi ↔ LoFi).
- **`dtype`** — weight/activation precision (`bf16 → bf8_b → bf4_b`; matmuls only).
- **`shard`** — memory sharding / L1 pinning / memory-config changes (this is the tensor-parallel/sharding lever).
- **`host`** — host- or dispatch-side work: **trace, fusion, gather, caching/KV-cache** (i.e. the structural/algorithmic rung is recorded here).
- **`tt-lang`** — a custom kernel authored in the tt-lang DSL.
- **`cpp`** — a custom C++ Metalium kernel.
- **`other`** — catch-all for any lever that doesn't classify into the columns above; rendered only when something lands there.

Cell legend: **`✓win`** = beat baseline (kept) · **`·try`** = measured, no gain · **`·wedge`** = wedged/crashed when tried · **`—`** = not attempted / not applicable.

> **Important — two ways it runs: trace mode vs eager mode.** Every profiled/verification run executes in one of two modes:
> - **Trace mode (the default).** The model's device command stream is captured once and replayed (`--enable_trace`), so timings reflect steady-state **on-device** performance with host/dispatch overhead removed. This is the fast path and the mode you want for real speed numbers.
> - **Eager mode (`TT_PERF_TRACE=0`).** Every op is dispatched one at a time on each call (`--disable_trace`) — slower and includes host overhead, but it's the mode to use when trace isn't supported for a model or when tracing perturbs correctness (i.e. accuracy over raw speed).
>
> Trace mode stays on by default. Toggle it with the **`TT_PERF_TRACE`** env var (`1` = trace, `0` = eager) — `optimize` has no `--no-trace` flag of its own (that flag belongs to the bring-up commands).

> **Important — one-time precondition for `optimize`.** For an existing model, `optimize` runs in a throwaway git **worktree**, which is a *clean checkout of your current branch* — it only sees **committed** files. So before you run it:
> 1. Be on the branch that has the **`tt_hw_planner` tool committed** (`scripts/tt_hw_planner/` **and** `models/experimental/perf_automation/`) — e.g. check out the tool's branch.
> 2. Make sure the **model's code is committed on that same branch** — bring the model baseline profile.
>
> If the tool or the model is only *uncommitted/untracked* on the branch you run from, the worktree won't contain it and the run fails before profiling. (This doesn't apply to `--in-place`, or to a model this tool brought up — those edit in place.)

**A · a model this tool brought up — just give the model id:**
```bash
python -m scripts.tt_hw_planner optimize <org>/<model> --devices all
```

**B · an existing tt-metal model — point at its code + PCC test:**
```bash
# code + tests in ONE folder — just give the folder:
python -m scripts.tt_hw_planner optimize models/demos/wormhole/bge_m3 --devices all

# code and tests in DIFFERENT folders — give both (the perf test is auto-generated from the PCC gate):
python -m scripts.tt_hw_planner optimize \
  --model-dir models/demos/bge_large_en \
  --pcc-test  models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py::test_ttnn_bge_model \
  --devices all
```

Options:
- `--devices single | 0,1 | all` — which chip(s). Default `0,1`; use **`all`** on a multi-chip board (a *partial* subset can trip a fabric error); use `single` on a one-chip machine.
- `--box` / `--mesh` — declared TT box + mesh shape (e.g. `--box p300c --mesh 2x2`) for **roofline calibration** — how close each op is to the hardware floor. **Note — this `--mesh` is not the bring-up one:** optimize uses it only for its **chip count**, then derives a kernel-viable **TP×DP** topology (`plan_parallelism` → `TP=cols`, `DP=rows`) rather than opening the literal physical arrangement.
- `--metric device_ms | wall_ms | auto` — what to optimize for (on-device time is the usual choice).
- `--in-place` — edit an existing demo's source directly instead of in a throwaway worktree (a tool-brought-up model is always in place).
- `--max-rounds N` — cc engine: max `claude -p` optimization rounds per pipeline (**default `3`**; one round is a full continuous agent session that climbs the whole ladder). Use `1` for a single pass, raise for models with lots of headroom; the deterministic gate can still stop earlier once each op is at its floor.
- `--target-band` / `--no-target-band` — **on by default.** Stop once the DRAM-bandwidth target band is reached (`IN_BAND → can_stop`), so a run ends at `min(band reached, --max-rounds)` — full-model uses the tok/s ceiling from active bytes, per-module each module's own roofline floor. Pass `--no-target-band` to keep optimizing past the band.
- `--module-level` — optimize graduated native modules **one at a time** (against each module's per-component PCC test) instead of the full pipeline — a coarse per-module pre-pass that sidesteps the heavy e2e baseline. Combine with `--modules a,b,c` to restrict to a subset, `--then-e2e` to run one full-pipeline pass afterward confirming the per-module wins survive composition, and `--reverify` to re-optimize modules already marked optimized in a prior run (default skips them, so a restart resumes at the next unoptimized module).
- `--hitl` — human-in-the-loop: the agent applies **one lever at a time**, then pauses at a block-level timing + rationale screen for your commit/revert/try decision before continuing (needs a live terminal).
- `--e2e-only` — cc engine: skip all optimization and just measure + print the full-model end-to-end time. Use to recover the before/after number if a prior run was stopped or killed before its final measurement.
- Advanced: `--perf-test path::test` (explicit perf test for models whose e2e overflows the profiler), `-k` / `--case` (pytest `-k` case override), and `--matmul-sweep` + `--matmul-sweep-pcc` / `--matmul-sweep-iters` / `--matmul-sweep-max-shapes` (a matmul fidelity×dtype pre-pass that writes a warm-start table; needs `--perf-test`).

> **Where edits land:** for an existing tt-metal demo, `optimize` runs in a **throwaway git worktree on a new branch** and leaves your files untouched — it prints how to `diff`/`merge` the kept speedups.

### Make the tool remember what it learned
Every verified speedup the agent improvises is **distilled back into a knob catalog** under `models/experimental/perf_automation/GUIDELINES/` — first as `LEARNED_*.md`, then promoted to `GRADUATED_*.md` once the same knob wins on a **second, different** model. On later runs the tool **recalls that catalog first** and reuses proven levers before improvising, so it gets faster across runs on its own. These catalog files ship with the tool, so **committing them is what makes the learning stick** — don't discard them.

To share learning across machines or teammates, opt into the shared catalog:
```bash
python -m scripts.tt_hw_planner optimize <org>/<model> --devices all --sync-catalog
```
`--sync-catalog` **pulls** the shared `GRADUATED_*` knobs from a catalog branch before the run and **pushes** newly-graduated ones after (`--catalog-remote`, default `origin`; `--catalog-branch`, default `perf-catalog`). Off by default — learning stays local unless you pass it.

## How to read what it prints
During bring-up you'll see lines like:
```
BRING-UP (cc) round 12: target=`encoder_layer` rung=emit  (graduated 6)
  | 03:12 | ✓ GRADUATED  | encoder_layer                | 6/24      | 18   |
  operations: 34/2862 on device (1%) ... on CPU (98%)
```
- **"graduated"** = that piece now runs on the chip and gives the right answers.
- **"on device vs on CPU"** = how much of the model is running on the Tenstorrent chip yet. This climbs toward 100% as it works.

You don't need to babysit it — just check back.

## If it stops with a message
The tool is designed to **stop and tell you the fix** rather than fail silently:

| Message says… | What to do |
|---|---|
| `import ttnn` failed | Run `./build_metal.sh` (step 2 above) |
| Not logged in / no API key | Do the `claude` login or set `ANTHROPIC_API_KEY` |
| Model is **gated** / 403 | Open the model page on HuggingFace, click "Request access", then `huggingface-cli login` |
| Can't **download weights** | Download on a machine with internet, copy them over, set `export HF_HOME=/your/copy` |
| Ran out of iterations, some pieces unfinished | Run `promote` (Section 1, Step 2) — it resumes only the leftovers |

## Glossary — terms you'll see
- **Component** = one piece of the model (like one layer).
- **Graduated** = that piece works correctly on the chip.
- **PCC** = the score that says "the chip's answer matches the original." Higher = better; it needs ~0.99 to pass.
- **Mesh / box** = your chip setup.
- **Optimize** = the speed-tuning stage (after the model already works).

## Where your results live
- The generated model: `models/demos/.../<model>/`
- Logs: inside that model's `_handoff/` folder.
- Speed-tuning results: `models/experimental/perf_automation/runs/<timestamp>/`

## Tips
- Long runs: use `tmux` or `nohup` so they survive an SSH disconnect.
- Want to watch everything live on screen? Prefix any command with `TT_HW_PLANNER_VERBOSE=1`.
- Registry drift check (maintenance): `python -m scripts.tt_hw_planner sync-registry --check` — exits non-zero if any mapped backend / building-block path is missing from the checkout (add `--no-unmapped` to skip the reverse "unmapped reusable module" hints).
