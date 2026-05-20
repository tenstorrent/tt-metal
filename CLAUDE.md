# Tenstorrent Model Bring-up Rules

## Environment
- ARCH_NAME=wormhole_b0
- PYTHONPATH must include $(pwd) and $(pwd)/models
- Env settings for every command run: export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

## Standards
- PCC > 0.99 is MANDATORY for all ttnn blocks.
- Reset device with 'tt-smi -r' if a hang occurs.
- Follow the "Relay Race" flow: Architecture -> Reference -> TTNN -> Debug -> Opt -> Server.
- Complete each part of the flow properly, in depth before moving to the next one.
- Critical: No shortcuts that need to be reverted later.
- Galaxy / multi-chip: weights for MLP, attention QKV, attention output, and large
  LM heads MUST use `ShardTensor2dMesh` (NOT `ReplicateTensorToMesh`). Only norm
  weights, RoPE tables, and small embedding tables may be replicated. Precedent:
  `models/demos/llama3_70b_galaxy/tt/llama_mlp.py`. Single-layer PCC tests will pass
  even when weights are wrongly replicated — verify per-device DRAM footprint at
  `__init__` time and load the full layer count before claiming the block is done.

## Parallelism assessment discipline (READ BEFORE ANY TP/DP DESIGN WORK)

Variable names lie. `num_device_groups`, `batch_size_per_device_group`,
`tt_data_parallel` — these can mean DP, batch routing, or KV-cache sharding
depending on the model. **Do not infer model parallelism from names, config
constants, or subagent summaries.**

Before claiming "model X uses TP=N / DP=M / 2D-TP / batch-axis routing",
verify with the **three-fact chain**:

1. **Weight dims** — `weight.shape` per chip after `ShardTensor2dMesh(dims=...)`.
   Write out the per-chip tensor shape explicitly.
2. **Matmul** — the `ttnn.linear` call that uses that weight. What's the
   input shape per chip? What's the output shape per chip? Is the output a
   complete result or a partial sum?
3. **CCL** — the `all_reduce` / `reduce_scatter` / `all_gather` immediately
   following. What `cluster_axis`? What ring size? What is it summing or
   gathering, in concrete terms?

If those three don't form a coherent matmul (e.g. partial K-sum → matching
col-axis all_reduce → complete output), the topology claim is **wrong** —
no matter how confident the source.

**A col-axis all_reduce(cluster_axis=1) is the signature of cols-as-TP-axis.**
A model with such a reduce CANNOT be DP across cols, regardless of how many
"device groups" the config defines. Cols-as-DP means **no col-axis CCL except
possibly a final logit gather at the LM head**.

PCC passing only means the math is internally consistent. It does NOT mean
the layout matches your mental model of what the layout is. Subagents
summarize and compress — their reports are hypotheses, not facts. Read the
actual code for any claim a downstream design depends on.

When two assessments disagree (yours now vs yours 30 min ago), STOP and
reconcile before proceeding. Don't rationalize past contradictions.

## Bring-up Flow

### Phase 1: Architecture
- Read ALL HF config JSON files and Python source files (not just config.json).
- Identify model family, parallelization plan, non-standard ops, masking mechanism.
- For VLMs: identify image token IDs (patches AND frame markers) from the HF processor.
- Produce ARCHITECTURE.md before writing any code.
- Skill: `/architecture`

### Phase 2: Reference
- Create standalone PyTorch implementations in `reference/functional.py`.
- Verify against HF model at each block level (PCC > 0.99).
- End-to-end functional verification (correct text/audio/image output), not just PCC.
- Skill: `/reference`

### Phase 3: TTNN
- Implement blocks in `tt/` using TTNN ops, achieving PCC > 0.99.
- T3K multi-chip: use `ttnn.all_reduce` (not `tt_all_reduce` which is a NO-OP on T3K).
- KV cache: use bfloat16 for vision models (bfloat8_b causes logit flips at S > 2500).
- Verify end-to-end output is functionally correct, not just PCC passing.
- Skill: `/ttnn`

### Phase 4: Debug
- Layer-by-layer PCC, device hangs, T3K-specific multi-chip bugs.
- Common T3K bugs: RoPE style mismatch, MLP gate/value order, TP QKV interleaving.
- Skill: `/debug`

### Phase 5: Optimization
- Tracy profiling: `python -m tracy -p -v -r profile_single_block.py --block <name>`
- Decode traces: do NOT use for servers with variable-length inputs (SDPA config baked at capture S).
- T3K TP MLP: use reduce_scatter + all_gather (trace-safe) not ttnn.all_reduce.
- Skill: `/optimization`

### Phase 6: tt-inference-server
- Expose model via OpenAI-compatible HTTP API through tt-inference-server.
- Skill: `/tt-inference-server`

## Completion Criteria

A model bring-up is **COMPLETE** when ALL of the following are satisfied:

### TTNN Block Quality
- [ ] All sub-block PCC tests pass (> 0.99 threshold)
- [ ] Full model end-to-end output is functionally correct vs HF reference

### Server Integration
- [ ] Server starts and handles requests without errors or device hangs
- [ ] Server accuracy ≥ demo accuracy (within 2-3 pp)
- [ ] 105-video (or equivalent) test suite run with results documented
- [ ] `tt-smi -r` not needed between requests

### Performance
- [ ] Tracy profiling completed for all sub-blocks
- [ ] Profile xlsx generated (`run_block_profiles.sh`)
- [ ] Dominant bottleneck identified and documented
- [ ] Latency vs GPU reference measured and documented

### Documentation
- [ ] BRINGUP_LOG.md updated with final PCC values, server results, performance numbers
- [ ] README.md created in `models/demos/{model}/` with: status, demo commands,
      server commands, test commands, profiling commands, known limitations
- [ ] All changes committed on feature branch

## Session Management
- Read BRINGUP_LOG.md at the start of every session.
- Update BRINGUP_LOG.md before ending a session with: [Status, PCC, Block Hash].

## Automated Bring-Up

The `/bringup` skill runs the full relay race autonomously.

```bash
/bringup https://huggingface.co/org/ModelName   # new model
/bringup ModelName                               # resume from Current Status
```

**What it does (unsupervised, escalates after 10 failed attempts per block):**
1. Repo-wide CPU audit across all existing models
2. Architecture: reads all HF configs + Python source, searches current `models/` by
   recency + device + arch similarity, looks up unit tests for every planned op
3. Reference: generates `reference/functional.py` with per-op capture, verifies PCC +
   element-wise p99 vs HF, saves persistent golden tensors at every op boundary
4. TTNN (serial, device occupied): adapts/generates blocks, per-block PCC test +
   integration test. Debug loop: locate failing op → isolation unit test → fix → log
5. End-to-end verification: runs ISL-matched prompt sets, compares TTNN vs reference
6. Server: generates generator_vllm.py from template, registers, runs test suite

**CPU rule**: `__init__` and preprocessing are fine; forward inference is never CPU.

**Debug loop**: 10 attempts per block → escalation report → STOP → wait for hint → reset counter.

**Log format**: `BRINGUP_LOG.md` is date-separated, append-only. `## Current Status` header
at top lets agent resume instantly without scanning the full log.

## Skills Available (individual phases)
- `/bringup` — full automated end-to-end bring-up (orchestrates all phases)
- `/architecture` — map HF model to TTNN blocks, parallelization plan
- `/reference` — create golden PyTorch reference for PCC verification
- `/ttnn` — implement TTNN blocks achieving PCC > 0.99
- `/debug` — diagnose PCC failures, device hangs, T3K multi-chip bugs
- `/optimization` — Tracy profiling, trace capture, CCL optimization
- `/tt-inference-server` — expose model via tt-inference-server OpenAI API

Your updates will be reviewed by Codex
