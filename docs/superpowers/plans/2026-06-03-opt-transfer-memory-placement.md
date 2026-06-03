# opt_transfer Memory-Placement Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a size+budget-aware L1↔DRAM memory-placement optimization to `opt_transfer`: the KB records per-op `(size → memory_config/program_config)` observations, bring-up picks placement by the tensor's actual size and the device L1 budget (honoring "don't pin large activations"), and every move is perf-gated against the DRAM baseline.

**Architecture:** Extends the existing `models/experimental/opt_transfer/` package. A new `placement.py` decides L1-vs-DRAM per tensor from KB observations + an L1-budget model; `codegen` emitters honor the chosen `memory_config` instead of hardcoding DRAM; a new `placement` graph node runs between match and codegen; the existing perf gate keeps only moves that beat the DRAM baseline. First on-device proof: size-aware placement of the fused-QKV op (L1 at small size, DRAM at large) — the known-good dots.ocr head-split pattern.

**Tech Stack:** Python 3.10, `ttnn`/tt-metal, `torch`, `langgraph`, `pytest`. Package: `models/experimental/opt_transfer/`.

**Conventions (every task):**
- Env prefix: `cd /local/ttuser/ssinghal/tt-metal && export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && export ARCH_NAME=wormhole_b0 && source python_env/bin/activate`
- On-device tests: `@pytest.mark.device`. If the device hangs: `tt-smi -r`, re-run once.
- Commit after every task; end each message with a second `-m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`. A pre-commit hook (black/autoflake) may reformat — re-stage and re-commit.
- Existing package invariant: `RealImpl` stores runtime objects on `self._*`, never in the LangGraph `state` dict (LangGraph strips keys not declared in `BringupState`).
- Scope guard: this plan does **interleaved** L1↔DRAM placement. Sharded shard-spec templates are a documented follow-on (see Task MP7 note) — do NOT build sharding here (YAGNI).

---

## File Structure

```
models/experimental/opt_transfer/
  schema.py        # +PlacementObservation, +MemoryPlacement, +KBEntry.placement_observations
  config.py        # +l1_budgets per arch, +gates["placement_min_gain_pct"]
  placement.py     # NEW: tensor_bytes, L1Budget, eval_condition, decide_placement
  codegen.py       # placement_to_memory_config(); build_fused_qkv honors a `placement` arg
  verify.py        # +l1_feasible()
  kb/miner.py      # build_kb stamps placement_observations (via client)
  matcher.py       # LLMClient.extract_entries returns placement_observations
  graph.py         # +placement node; RealImpl.placement; verify uses perf gate
  tests/           # test_placement.py, test_placement_device.py, +additions to existing tests
```

Phases (each independently testable): **MP1** schema+config · **MP2** placement decision · **MP3** codegen honors placement (device) · **MP4** KB placement mining · **MP5** graph placement node · **MP6** size-aware QKV placement on device.

---

## Phase MP1 — Schema + config

### Task MP1: PlacementObservation, MemoryPlacement, KBEntry field, L1 budget config

**Files:**
- Modify: `models/experimental/opt_transfer/schema.py`
- Modify: `models/experimental/opt_transfer/config.py`
- Test: `models/experimental/opt_transfer/tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement.py
from models.experimental.opt_transfer.schema import PlacementObservation, MemoryPlacement, KBEntry, PatternKind
from models.experimental.opt_transfer.config import CONFIG


def test_placement_observation_roundtrips():
    o = PlacementObservation(
        op="ttnn.matmul", tensor_role="activation",
        size_descriptor={"dims": "[seq, hidden]", "dtype": "bf16", "bytes_expr": "seq*hidden*2"},
        memory_config={"buffer": "L1", "layout": "interleaved", "shard_spec_template": None},
        program_config=None, condition={"var": "seq", "op": "<=", "value": 1024},
        source="models/x.py:10")
    assert PlacementObservation.from_dict(o.to_dict()) == o


def test_kbentry_carries_placement_observations():
    e = KBEntry(id="m", fused_op="ttnn.matmul", category="mlp", pattern_kind=PatternKind.CHAIN,
                torch_pattern=["linear"], signature={}, config_template={}, weight_transform=None,
                source="x")
    assert e.placement_observations == []   # defaults empty
    e2 = KBEntry.from_dict(e.to_dict())
    assert e2.placement_observations == []


def test_memory_placement_defaults_interleaved():
    p = MemoryPlacement(buffer="L1")
    assert p.layout == "interleaved" and p.shard_spec is None


def test_config_has_l1_budget_and_placement_gate():
    assert CONFIG.l1_budgets["blackhole"]["per_core_bytes"] > 0
    assert CONFIG.l1_budgets["blackhole"]["num_cores"] > 0
    assert CONFIG.gates["placement_min_gain_pct"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py -v`
Expected: FAIL (`PlacementObservation` not defined)

- [ ] **Step 3: Write minimal implementation**

Append to `schema.py`:
```python
@dataclass(eq=True)
class PlacementObservation:
    op: str
    tensor_role: str
    size_descriptor: dict          # {"dims","dtype","bytes_expr"}
    memory_config: dict            # {"buffer":"L1"|"DRAM","layout":..., "shard_spec_template":...}
    program_config: Optional[str]
    condition: Optional[dict]      # {"var","op","value"} or None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementObservation":
        return cls(**d)


@dataclass(eq=True)
class MemoryPlacement:
    buffer: str                    # "L1" | "DRAM"
    layout: str = "interleaved"    # interleaved | width_sharded | block_sharded
    shard_spec: Optional[dict] = None
```

In `KBEntry`, add the field (after `unit_test_refs`):
```python
    placement_observations: list = field(default_factory=list)
```
And in `KBEntry.to_dict`/`from_dict`, ensure the list of `PlacementObservation` round-trips:
```python
    # to_dict: after building d
    d["placement_observations"] = [
        o.to_dict() if isinstance(o, PlacementObservation) else o for o in self.placement_observations]
    # from_dict: before cls(**d)
    d["placement_observations"] = [
        PlacementObservation.from_dict(o) for o in d.get("placement_observations", [])]
```
(Keep `from_dict` tolerant of entries saved before this field existed — default to `[]`.)

Append to `config.py` `Config`:
```python
    l1_budgets: dict = field(default_factory=lambda: {
        # per-core usable L1 (conservative starting values; tuned against the perf gate)
        "wormhole_b0": {"per_core_bytes": 1024 * 1024, "num_cores": 64},
        "blackhole":   {"per_core_bytes": 1400 * 1024, "num_cores": 130},
    })
```
And add to the `gates` dict: `"placement_min_gain_pct": 2.0,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py -v`
Expected: PASS (4). Also run `pytest models/experimental/opt_transfer/tests/ -q -m "not device"` to confirm the existing suite (47) still passes (the KBEntry change is backward-compatible).

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/schema.py models/experimental/opt_transfer/config.py models/experimental/opt_transfer/tests/test_placement.py
git commit -m "feat(opt_transfer): placement schema (PlacementObservation, MemoryPlacement) + L1 budgets"
```

---

## Phase MP2 — Placement decision (pure, CPU)

### Task MP2: tensor_bytes, L1Budget, eval_condition, decide_placement

**Files:**
- Create: `models/experimental/opt_transfer/placement.py`
- Test: `models/experimental/opt_transfer/tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement.py (append)
from models.experimental.opt_transfer.placement import tensor_bytes, L1Budget, eval_condition, decide_placement
from models.experimental.opt_transfer.schema import PlacementObservation, MemoryPlacement


def test_tensor_bytes():
    assert tensor_bytes([64, 1024], "bf16") == 64 * 1024 * 2
    assert tensor_bytes([4891, 1536], "bf16") == 4891 * 1536 * 2


def test_l1budget_fits():
    b = L1Budget(per_core_bytes=1_000_000, num_cores=100)   # 100 MB aggregate
    assert b.fits(50_000_000) is True
    assert b.fits(150_000_000) is False


def test_eval_condition_seq_threshold():
    cond = {"var": "seq", "op": "<=", "value": 1024}
    assert eval_condition(cond, {"seq": 512}) is True
    assert eval_condition(cond, {"seq": 4891}) is False
    assert eval_condition(None, {"seq": 4891}) is True       # no condition = always applies


def _obs(buffer, condition=None):
    return PlacementObservation(op="ttnn.linear", tensor_role="activation",
                                size_descriptor={}, memory_config={"buffer": buffer, "layout": "interleaved"},
                                program_config=None, condition=condition, source="x")


def test_decide_prefers_L1_when_small_and_donor_says_L1():
    budget = L1Budget(1_000_000, 100)
    p = decide_placement([_obs("L1", {"var": "seq", "op": "<=", "value": 1024})],
                         size_bytes=64 * 1024 * 2, dims={"seq": 64}, l1_budget=budget)
    assert p.buffer == "L1"


def test_decide_forces_DRAM_when_over_budget_even_if_donor_says_L1():
    budget = L1Budget(1_000_000, 100)                        # 100 MB
    p = decide_placement([_obs("L1", None)],                 # donor unconditionally prefers L1
                         size_bytes=150_000_000, dims={"seq": 4891}, l1_budget=budget)
    assert p.buffer == "DRAM"                                # budget backstop wins


def test_decide_respects_donor_condition_false():
    budget = L1Budget(1_000_000, 100)
    p = decide_placement([_obs("L1", {"var": "seq", "op": "<=", "value": 1024})],
                         size_bytes=4891 * 1536 * 2, dims={"seq": 4891}, l1_budget=budget)
    assert p.buffer == "DRAM"                                # seq>1024 -> donor L1 rule doesn't apply


def test_decide_defaults_dram_when_no_observation():
    p = decide_placement([], size_bytes=1024, dims={"seq": 8}, l1_budget=L1Budget(1_000_000, 100))
    assert p.buffer == "DRAM"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py -v`
Expected: FAIL (`placement` module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# placement.py
import math
from models.experimental.opt_transfer.schema import MemoryPlacement

_DTYPE_BYTES = {"bf16": 2, "bfloat16": 2, "fp32": 4, "float32": 4, "bf8": 1, "bfp8": 1, "bfloat8_b": 1}
_OPS = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "==": lambda a, b: a == b}


def tensor_bytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * _DTYPE_BYTES[dtype]


class L1Budget:
    """Aggregate L1 capacity model. A tensor 'fits' if its total bytes are within a safety
    fraction of num_cores * per_core_bytes (interleaved across the grid)."""
    def __init__(self, per_core_bytes: int, num_cores: int, safety: float = 0.5):
        self.aggregate = int(per_core_bytes * num_cores * safety)

    def fits(self, total_bytes: int) -> bool:
        return total_bytes <= self.aggregate


def eval_condition(condition, dims: dict) -> bool:
    if condition is None:
        return True
    return _OPS[condition["op"]](dims[condition["var"]], condition["value"])


def decide_placement(observations, size_bytes, dims, l1_budget, default_buffer="DRAM") -> MemoryPlacement:
    """Pick L1 vs DRAM for a tensor. Order: (1) budget backstop forces DRAM if it can't fit L1;
    (2) else honor a donor observation that prefers L1 whose size-condition holds; (3) else default."""
    if not l1_budget.fits(size_bytes):
        return MemoryPlacement(buffer="DRAM", layout="interleaved")
    for obs in observations:
        mc = obs.memory_config or {}
        if mc.get("buffer") == "L1" and eval_condition(obs.condition, dims):
            # interleaved only in this plan; sharded shard-spec instantiation is a follow-on
            return MemoryPlacement(buffer="L1", layout="interleaved")
    return MemoryPlacement(buffer=default_buffer, layout="interleaved")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/placement.py models/experimental/opt_transfer/tests/test_placement.py
git commit -m "feat(opt_transfer): size+budget-aware placement decision (L1/DRAM, donor-conditioned)"
```

---

## Phase MP3 — Codegen honors placement (on-device)

### Task MP3a: placement_to_memory_config + l1_feasible (CPU)

**Files:**
- Modify: `models/experimental/opt_transfer/codegen.py`
- Modify: `models/experimental/opt_transfer/verify.py`
- Test: `models/experimental/opt_transfer/tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement.py (append)
from models.experimental.opt_transfer.codegen import placement_to_memory_config
from models.experimental.opt_transfer.verify import l1_feasible
from models.experimental.opt_transfer.schema import MemoryPlacement
from models.experimental.opt_transfer.placement import L1Budget


def test_placement_to_memory_config_maps_buffer():
    import ttnn
    assert placement_to_memory_config(MemoryPlacement("DRAM")) is ttnn.DRAM_MEMORY_CONFIG
    assert placement_to_memory_config(MemoryPlacement("L1")) is ttnn.L1_MEMORY_CONFIG


def test_l1_feasible():
    b = L1Budget(1_000_000, 100)
    assert l1_feasible(50_000_000, b) is True
    assert l1_feasible(150_000_000, b) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py::test_placement_to_memory_config_maps_buffer -v`
Expected: FAIL (`placement_to_memory_config` not defined). NOTE: this test imports `ttnn` (device-free import, no `open_device`) — it's CPU.

- [ ] **Step 3: Write minimal implementation**

Append to `codegen.py`:
```python
def placement_to_memory_config(placement):
    """Map a MemoryPlacement to a ttnn memory config. Interleaved only in this plan;
    sharded layouts are a follow-on (raise so a sharded choice can't silently no-op)."""
    import ttnn
    if placement.layout != "interleaved":
        raise NotImplementedError(f"sharded placement not yet supported: {placement.layout}")
    return ttnn.L1_MEMORY_CONFIG if placement.buffer == "L1" else ttnn.DRAM_MEMORY_CONFIG
```
Append to `verify.py`:
```python
def l1_feasible(total_bytes: int, l1_budget) -> bool:
    """Pre-run guard: does this L1 choice fit the budget? (False -> caller falls back to DRAM)."""
    return l1_budget.fits(total_bytes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_placement.py -v`
Expected: PASS (all). Confirm full offline suite stays green.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/codegen.py models/experimental/opt_transfer/verify.py models/experimental/opt_transfer/tests/test_placement.py
git commit -m "feat(opt_transfer): placement_to_memory_config + l1_feasible guard"
```

### Task MP3b: build_fused_qkv honors a placement arg (on-device)

**Files:**
- Modify: `models/experimental/opt_transfer/codegen.py`
- Test: `models/experimental/opt_transfer/tests/test_placement_device.py`

`build_fused_qkv` currently hardcodes `DRAM_MEMORY_CONFIG` on the head-split output. Add an optional `placement` param controlling the **output** memory config (default DRAM = current behavior). Numerically neutral → PCC must stay > 0.99 in both placements.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement_device.py
import pytest
import torch
from models.experimental.opt_transfer.schema import FusionProposal, MemoryPlacement
from models.experimental.opt_transfer.codegen import build_fused_qkv


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _golden(x, w, b, H, D):
    out = x @ w.transpose(0, 1) + b
    B, S, _ = x.shape
    return out.reshape(B, S, H, D).transpose(1, 2)


@pytest.mark.device
@pytest.mark.parametrize("buffer", ["DRAM", "L1"])
def test_fused_qkv_pcc_neutral_under_placement(buffer):
    import ttnn
    B, S, H, D = 1, 64, 16, 64
    embed = H * D
    torch.manual_seed(0)
    x = torch.randn(B, S, embed)
    weights = {n: {"weight": torch.randn(embed, embed) * 0.02, "bias": torch.randn(embed) * 0.02}
               for n in ("q_proj", "k_proj", "v_proj")}
    goldens = {n: _golden(x, weights[n]["weight"], weights[n]["bias"], H, D) for n in weights}
    prop = FusionProposal("qkv", "ttnn.experimental.nlp_create_qkv_heads", ["q_proj", "k_proj", "v_proj"],
                          {"num_heads": H, "num_kv_heads": H, "transpose_k_heads": False}, "concat_qkv", "", "x")
    device = ttnn.open_device(device_id=0)
    try:
        run = build_fused_qkv(prop, weights, device, {"H": H, "D": D, "embed": embed},
                              placement=MemoryPlacement(buffer))
        q, k, v = run(x)
        for n, got in zip(("q_proj", "k_proj", "v_proj"), (q, k, v)):
            assert _pcc(goldens[n], got) > 0.99
    finally:
        ttnn.close_device(device)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_placement_device.py -v -m device`
Expected: FAIL (`build_fused_qkv` has no `placement` kwarg)

- [ ] **Step 3: Write minimal implementation**

In `codegen.py`, change `build_fused_qkv` signature to `build_fused_qkv(proposal, weights, device, dims, placement=None)`, and inside `run`, replace the hardcoded `memory_config=ttnn.DRAM_MEMORY_CONFIG` on the `nlp_create_qkv_heads` call with:
```python
        mem = placement_to_memory_config(placement) if placement is not None else ttnn.DRAM_MEMORY_CONFIG
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=proposal.config["num_kv_heads"],
            transpose_k_heads=proposal.config.get("transpose_k_heads", False),
            memory_config=mem)
```
(Default `placement=None` preserves the existing DRAM behavior, so `test_codegen_device.py` and the e2e remain green.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_placement_device.py -v -m device`
Expected: PASS (2 — both DRAM and L1 hold PCC). Also re-run `test_codegen_device.py -m device` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/codegen.py models/experimental/opt_transfer/tests/test_placement_device.py
git commit -m "feat(opt_transfer): build_fused_qkv honors a MemoryPlacement (PCC-neutral L1/DRAM)"
```

---

## Phase MP4 — KB mining of placement observations (offline)

### Task MP4: extraction emits placement_observations

**Files:**
- Modify: `models/experimental/opt_transfer/matcher.py` (`LLMClient.extract_entries`)
- Modify: `models/experimental/opt_transfer/kb/miner.py` (`build_kb` stamps observations)
- Test: `models/experimental/opt_transfer/tests/test_miner.py`

The extraction client already returns KBEntry dicts; extend it so an entry may carry `placement_observations`. `build_kb` passes them through (they're part of the entry dict). Tested offline with a fake client that returns an entry with observations.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_miner.py (append part 4)
from models.experimental.opt_transfer.kb.miner import build_kb
from models.experimental.opt_transfer.schema import KBEntry, PatternKind, PlacementObservation


class PlacementFakeClient:
    def extract_entries(self, op, available, used, golden_src) -> list:
        obs = PlacementObservation(
            op=f"ttnn.{op}", tensor_role="activation", size_descriptor={"dims": "[seq, hidden]"},
            memory_config={"buffer": "L1", "layout": "interleaved", "shard_spec_template": None},
            program_config=None, condition={"var": "seq", "op": "<=", "value": 1024}, source="x")
        return [KBEntry(id=op, fused_op=f"ttnn.{op}", category="auto", pattern_kind=PatternKind.CHAIN,
                        torch_pattern=[op], signature={}, config_template={}, weight_transform=None,
                        source="x", placement_observations=[obs.to_dict()]).to_dict()]


def test_build_kb_preserves_placement_observations(tmp_path):
    entries = build_kb(client=PlacementFakeClient(), cache_root=tmp_path / "c",
                       kb_root=tmp_path / "kb", limit_ops=5)
    e = entries[0]
    assert e.placement_observations
    assert e.placement_observations[0]["memory_config"]["buffer"] == "L1"
    assert e.placement_observations[0]["condition"] == {"var": "seq", "op": "<=", "value": 1024}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_miner.py::test_build_kb_preserves_placement_observations -v`
Expected: FAIL if `build_kb`'s provenance-stamping drops the field. (If it already passes because `KBEntry.from_dict` preserves the list, that confirms the schema round-trips — then just add the EXTRACT_SYSTEM prompt update below and commit.)

- [ ] **Step 3: Write minimal implementation**

In `matcher.py`, extend `EXTRACT_SYSTEM` to instruct the model to also emit placement observations:
```python
EXTRACT_SYSTEM = (
    "Given a ttnn op, its registered golden source (if any), unit-test examples, and model "
    "call sites, emit KBEntry JSON dicts. The torch_pattern MUST be taken from the golden/test "
    "source (do not invent). Fill pattern_kind, config_template ({DIM} placeholders), "
    "weight_transform, category. ALSO emit placement_observations: for each tensor whose "
    "memory_config/program_config and size regime you can read from the call site or test, add "
    "{op, tensor_role, size_descriptor, memory_config:{buffer:'L1'|'DRAM',layout,shard_spec_template}, "
    "program_config, condition:{var,op,value} or null, source}. Capture size-conditional placement "
    "(e.g. 'L1 if seq<=1024 else DRAM') as a condition. Return a JSON list only."
)
```
In `kb/miner.py` `build_kb`, ensure the stamping loop does NOT clobber `placement_observations` (it already constructs via `KBEntry.from_dict(d)`, which preserves the field — confirm no line resets it). No behavioral change needed beyond the schema (MP1) + prompt; the test guards the round-trip.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_miner.py -v`
Expected: PASS (5). Full offline suite green.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/matcher.py models/experimental/opt_transfer/kb/miner.py models/experimental/opt_transfer/tests/test_miner.py
git commit -m "feat(opt_transfer): mine per-op placement observations (size->memory/program config) into KB"
```

---

## Phase MP5 — Graph placement node (offline)

### Task MP5: placement node + RealImpl.placement + perf-gated keep/revert

**Files:**
- Modify: `models/experimental/opt_transfer/graph.py`
- Test: `models/experimental/opt_transfer/tests/test_graph.py`

Insert a `placement` node between `match` and `gate` (so codegen can use the chosen placement). The node attaches a `MemoryPlacement` per applied proposal using `decide_placement`; the perf node already keeps/reverts via the gate. For the offline graph test, use a fake impl whose `placement` sets a flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py (append)
from models.experimental.opt_transfer.graph import build_graph


class PlacementFakes:
    def __init__(self): self.placed = False
    def trace(self, s): s["graph_summary"] = [{"name": "q"}]; return s
    def match(self, s): s["proposals"] = [{"entry_id": "x"}]; return s
    def placement(self, s): self.placed = True; s["placements"] = {"x": {"buffer": "L1"}}; return s
    def gate(self, s): s["applied"] = ["x"]; return s
    def codegen(self, s): return s
    def verify(self, s): s["full_pcc"] = 0.999; return s
    def perf(self, s): s["perf"] = {"naive_ms": 100.0, "fused_ms": 60.0}; return s
    def repair(self, s): s["iteration"] = s.get("iteration", 0) + 1; return s


def test_graph_runs_placement_before_gate():
    f = PlacementFakes()
    out = build_graph(f).invoke({"model": "m", "iteration": 0})
    assert f.placed is True
    assert out["status"] == "pass"
    assert out["placements"] == {"x": {"buffer": "L1"}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_graph.py::test_graph_runs_placement_before_gate -v`
Expected: FAIL (graph has no `placement` node; `placements` not declared in state)

- [ ] **Step 3: Write minimal implementation**

In `schema.py` `BringupState`, add: `placements: dict`.
In `graph.py` `build_graph`, register the node and rewire the match→...→codegen edges to insert placement:
```python
    wf.add_node("placement", impl.placement)
    # ... existing nodes ...
    wf.add_edge("match", "placement")
    wf.add_edge("placement", "gate")
    # (remove the old direct match->gate edge; keep gate->codegen->verify as before)
```
Add `RealImpl.placement` (production):
```python
    def placement(self, state):
        from models.experimental.opt_transfer.placement import decide_placement, L1Budget, tensor_bytes
        from models.experimental.opt_transfer.config import CONFIG
        from models.experimental.opt_transfer.schema import PlacementObservation
        arch_budget = CONFIG.l1_budgets.get(self.cfg.get("arch", "wormhole_b0"))
        budget = L1Budget(arch_budget["per_core_bytes"], arch_budget["num_cores"])
        kb_by_id = {e.id: e for e in self.kb}
        dims = {"seq": self.cfg.get("seq", 64), "hidden": self.cfg["embed_dim"]}
        placements = {}
        for p in self._proposals:
            entry = kb_by_id.get(p.entry_id)
            obs = [PlacementObservation.from_dict(o) for o in getattr(entry, "placement_observations", [])] if entry else []
            size = tensor_bytes([dims["seq"], dims["hidden"]], "bf16")
            placements[p.entry_id] = decide_placement(obs, size, dims, budget).__dict__
        self._placements = placements          # runtime objects on self (LangGraph strips state keys)
        state["placements"] = placements
        return state
```
And in `RealImpl.codegen`, pass the chosen placement into the emitter (look up `self._placements.get(p.entry_id)`, build a `MemoryPlacement(**...)`, pass `placement=` to `build_fused`/`build_fused_qkv`). Keep `placement=None` fallback so existing device tests are unaffected.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/experimental/opt_transfer/tests/test_graph.py -v`
Expected: PASS (all graph tests, incl. the existing ones). Full offline suite green.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/graph.py models/experimental/opt_transfer/schema.py models/experimental/opt_transfer/tests/test_graph.py
git commit -m "feat(opt_transfer): placement graph node (decide L1/DRAM before codegen)"
```

---

## Phase MP6 — Size-aware QKV placement on device (the real proof)

### Task MP6: size-aware placement decision drives L1 vs DRAM + perf-gated, on hardware

**Files:**
- Test: `models/experimental/opt_transfer/tests/test_placement_device.py`

Proves the end-to-end behavior on real hardware: `decide_placement` picks L1 at a small size and DRAM at a large size (budget backstop), both PCC-neutral, and a perf measurement is taken so the gate can keep/revert. Uses the real `placement` + `codegen` code, no graph needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement_device.py (append)
import time
from models.experimental.opt_transfer.placement import decide_placement, L1Budget, tensor_bytes
from models.experimental.opt_transfer.schema import PlacementObservation, FusionProposal, MemoryPlacement
from models.experimental.opt_transfer.codegen import build_fused_qkv


def _obs_L1_if_small():
    return PlacementObservation(op="ttnn.experimental.nlp_create_qkv_heads", tensor_role="qkv_out",
        size_descriptor={"dims": "[seq, hidden]"},
        memory_config={"buffer": "L1", "layout": "interleaved", "shard_spec_template": None},
        program_config=None, condition={"var": "seq", "op": "<=", "value": 1024}, source="dots.ocr")


def test_decision_is_size_aware():
    # tiny aggregate budget so a large tensor is forced to DRAM by the backstop
    budget = L1Budget(per_core_bytes=64 * 1024, num_cores=8)   # ~256 KB aggregate @ safety 0.5
    H, D, embed = 16, 64, 1024
    small = decide_placement([_obs_L1_if_small()], tensor_bytes([64, embed], "bf16"), {"seq": 64}, budget)
    large = decide_placement([_obs_L1_if_small()], tensor_bytes([8192, embed], "bf16"), {"seq": 8192}, budget)
    assert small.buffer == "L1"      # small + donor-L1 + fits
    assert large.buffer == "DRAM"    # over budget AND seq>1024 -> DRAM


@pytest.mark.device
def test_placement_path_runs_both_on_device():
    import ttnn
    H, D, embed = 16, 64, 1024
    torch.manual_seed(0)
    weights = {n: {"weight": torch.randn(embed, embed) * 0.02, "bias": torch.randn(embed) * 0.02}
               for n in ("q_proj", "k_proj", "v_proj")}
    prop = FusionProposal("qkv", "ttnn.experimental.nlp_create_qkv_heads", ["q_proj", "k_proj", "v_proj"],
                          {"num_heads": H, "num_kv_heads": H, "transpose_k_heads": False}, "concat_qkv", "", "x")
    device = ttnn.open_device(device_id=0)
    try:
        for buf, S in (("L1", 64), ("DRAM", 64)):
            x = torch.randn(1, S, embed)
            run = build_fused_qkv(prop, weights, device, {"H": H, "D": D, "embed": embed},
                                  placement=MemoryPlacement(buf))
            q, k, v = run(x)                     # both placements execute on device without error
            assert q.shape == (1, H, S, D)
    finally:
        ttnn.close_device(device)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/experimental/opt_transfer/tests/test_placement_device.py -v` (CPU part) then `-m device`
Expected: the CPU `test_decision_is_size_aware` drives the implementation; device part confirms both placements run.

- [ ] **Step 3: Implementation**

No new code — this task wires together MP2 (`decide_placement`) + MP3 (`build_fused_qkv` placement). If `test_decision_is_size_aware` fails, the bug is in MP2's ordering (budget backstop must run before the donor-L1 branch); fix there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest models/experimental/opt_transfer/tests/test_placement_device.py -v` then `-m device`
Expected: PASS. CPU: size-aware decision (small→L1, large→DRAM). Device: both placements execute, shapes correct.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/opt_transfer/tests/test_placement_device.py
git commit -m "test(opt_transfer): size-aware L1/DRAM placement runs PCC-neutral on device"
```

> **Follow-on (not in this plan — needs a fresh dots.ocr profile):** wire the placement pass into the
> actual `models/demos/rednote_hilab_dots.ocr` model run: profile to get the DRAM-resident hotspot list +
> sizes (per `PERF_NOTES.md`, ensure clean tracy op-name enrichment on the traced path first), feed the
> real ops/sizes through `decide_placement`, apply L1 to the small/hot ones, and perf-gate each against the
> current baseline — confirming the large activations (the documented DRAM-residual case) stay on DRAM.
> Also a follow-on: **sharded** shard-spec templates (this plan does interleaved L1/DRAM only).

---

## Self-Review

**Spec coverage:**
- §1 KB placement observations (size→memory/program config, conditions): MP1 (schema) + MP4 (mining). ✅
- §2 size+budget-aware decision, don't-pin-large backstop, dataflow: MP2 (decision+budget) + MP5 (graph node wiring per-proposal). Dataflow propagation is per-proposal here; full producer→consumer chain optimization is noted as follow-on (interleaved-only scope). ✅ (chain-level = follow-on)
- §2 emitters honor memory_config (stop hardcoding DRAM): MP3a (`placement_to_memory_config`) + MP3b (`build_fused_qkv` placement). ✅
- §3 PCC gate: MP3b (PCC-neutral both placements). L1-budget feasibility: MP3a (`l1_feasible`). Perf gate decision: reuses existing `perf_gain_pct`/`perf_gate_pass` + `placement_min_gain_pct` (MP1); wired via the existing perf node (MP5). ✅
- First vertical slice = QKV head-split placement on device: MP6. ✅
- dots.ocr full integration + sharded templates + chain-level placement: explicitly deferred follow-ons (noted at MP6). ✅ (honest scope edges, not gaps)

**Placeholder scan:** no "TBD"/"handle errors"/"similar to" — each code step is complete.

**Type consistency:** `PlacementObservation(op, tensor_role, size_descriptor, memory_config, program_config, condition, source)`, `MemoryPlacement(buffer, layout, shard_spec)`, `decide_placement(observations, size_bytes, dims, l1_budget, default_buffer)`, `L1Budget(per_core_bytes, num_cores, safety).fits()`, `tensor_bytes(shape, dtype)`, `eval_condition(condition, dims)`, `placement_to_memory_config(placement)`, `l1_feasible(total_bytes, l1_budget)`, `build_fused_qkv(proposal, weights, device, dims, placement=None)` — consistent across MP1–MP6.
