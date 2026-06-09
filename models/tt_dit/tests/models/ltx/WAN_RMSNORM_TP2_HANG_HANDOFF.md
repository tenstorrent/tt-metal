# Handoff: `tp2_a_selfattn_qk` hang in fused distributed RMSNorm (WH Galaxy)

**Branch:** `cglagovich/fused_rms_norm` (HEAD `d507ed3c056`)
**Op:** `WanFusedDistributedRmsnormDeviceOperation`
**Status:** root cause localized to a single LLK call; fix not yet landed.

---

## 1. What we're debugging

The fused distributed RMSNorm device op (`ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/`)
hangs deterministically for the LTX **audio self-attention QK-norm** config at **TP=2**:

| field | value |
|---|---|
| config id | `tp2_a_selfattn_qk` |
| shape | rows=64 (2 tile-rows), feat=1024 (32 tile-cols), 16 heads, head_dim=64 |
| flags | `fuse_rope=1`, `per_head_rope=1`, `per_head_norm=0` (whole-row norm), `has_weight=1` |
| ring | TP=2 → `ring_size=2`, 1×2 LINE submesh (devices **24 & 25**) |
| chunking | per-head-rope forces `chunk_size_rows=1` → **2 chunks** (this is the trigger, see below) |

**Trigger (important):** the hang requires **`ring_size==2` AND multi-chunk (`chunk_size_rows==1`)**.
- `tp2_a` at *natural* `chunk_size_rows=2` (single chunk, 2 rows) **RUNS** (~73 µs).
- `chunk_size_rows=1` (2 chunks, forced by the per-head-rope L1 clamp) **HANGS**.
- TP=1 with the identical compute shape **PASSES** (regression test
  `test_wan_fused_distributed_rmsnorm_tp1_rope`, param `N64_H1024_16heads_perhead`).

So per-head-rope only matters because it forces `chunk=1`; the real axis is **multi-chunk × ring_size=2**.

---

## 2. Root cause — localized to ONE call

The **compute kernel wedges in the POST phase `P_NMUL` init**, specifically at:

```
ttnn/.../device/kernels/compute/wan_rmsnorm_fused_compute.cpp  (~line 393)
    reconfig_data_format(input_cb, reduce_result_cb);     // completes
    pack_reconfig_data_format(mul_rms_result_cb);          // <-- HANGS HERE
    mul_bcast_cols_init_short(input_cb, reduce_result_cb);
```

`pack_reconfig_data_format(new_cb)` → `PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb)))`
(`tt_metal/hw/inc/api/compute/pack.h:132`) — a **PACK-thread** (TRISC2) packer reconfig.

This happens **immediately after the matmul-based `reduce<AVG,REDUCE_ROW>`** (which packs `reduce_result`).
The reduce itself **completes** (reduce_result is produced and confirmed available). Then PACK wedges
in the reconfig → MATH stuck at `MWDD` (math-wait-dest, can't get dest from the stalled packer) →
the writer (`CWFW`, output_cb empty) and reader (`CRBW`, rope_cos_cb full from row-0 never consumed)
are **downstream victims**, not the cause.

### How we know (evidence)
- **Watcher waypoints** for the hung worker (Device 24 & 25, `worker core(0,0)`, `h_id:9`, symmetric):
  `BRISC=CWFW, NCRISC=CRBW, TRISC0(unpack)=UPMD, TRISC1(math)=MWDD, TRISC2(pack)=K`.
  (`UPMD` = matmul-unpack waypoint, from `llk_unpack_AB_matmul_api.h`.)
- **Watcher ring buffer** breadcrumbs (`WATCHER_RING_BUFFER_PUSH`) in the compute POST, read live while
  hung. Final progress (newest→oldest): `[0xC0131000, 0xC0130500, 0xC0130000]`:
  - `0xC0130000` = reduce done (`cb_wait_front(reduce_result_cb,1)` returned)
  - `0xC0130500` = entered P_NMUL
  - `0xC0131000` = **after `reconfig_data_format`**
  - **never reaches `0xC0132000`** (after `pack_reconfig_data_format`) ⇒ wedged in that call.
- tt-triage confirmed: NOT a NoC-hardware / fabric / eth / NoC-command-buffer hang (those checks pass
  on the op cores); the one `check_noc_status` failure is on stale *neighbor* cores (red herring).

### Disproven hypotheses (each tested on HW; none fixed the hang)
1. Cross-chip all-gather / fabric deadlock — refuted (writer is a victim at `CWFW`, not stuck in AG).
2. Output `output_cb` back-pressure — deepening 2×→4× did nothing.
3. `stats_gathered_cb` depth — deepening 1×→2× did nothing.
4. Reduce CB-pointer duplication — audited: `reduce<>` uses `WaitAndPopPerTile` and correctly manages
   its CBs (wait+pop input `stats_gathered`, reserve+push output `reduce_result`); the compute does NOT
   duplicate (AGWAIT is a non-consuming wait, reduce_result is consume-only). Balanced & correctly sized.
5. Reduce input policy `BulkWaitBulkPop` instead of default — still hangs.
6. Single-tile reduce `ReduceInputBlockShape::row(1u)` instead of `row(stats_tiles_cols)` — still hangs.
   ⇒ the reduce tile-count / CB-policy is **not** the cause.
7. `reduce_uninit` skip: `reduce<>` skips `reduce_uninit` when `use_matmul`
   (`reduce_helpers_compute.inl:535`). But `use_matmul = reduce_uses_matmul<AVG,REDUCE_ROW>()` is
   compile-time and identical for TP=1 (which passes), so the skip alone isn't the differentiator.

---

## 3. Build & run on WH Galaxy

Host: WH 4×8 Galaxy. Repo at `/home/cglagovich/tt-metal`. Sibling dirs: `../setup_env.sh`,
`../smi_venv/` (tt-smi), `../<...>` toolchains. C++ host is prebuilt in `build_Release/`.
**Kernels are JIT-compiled at runtime** — editing a `.cpp` kernel does NOT require a host rebuild,
just rerun (first run after an edit pays a cold compile, ~60–120 s).

### Environment
```bash
cd /home/cglagovich/tt-metal
source ../setup_env.sh                 # sets TT_METAL_HOME etc.
source python_env/bin/activate
export PATH="$HOME/.local/bin:$PATH"    # for uv
```

### Device reset (FLAKY — budget for retries)
`tt-smi -glx_reset` fails `POST_RESET` for a random device ~50% of the time. Retry 2–5×:
```bash
for i in 1 2 3 4 5; do
  r=$(../smi_venv/bin/tt-smi -glx_reset 2>&1 | grep -iE "Re-initialized|POST_RESET failed" | tail -1)
  echo "reset $i: $r"; echo "$r" | grep -q "Re-initialized" && break; sleep 3
done
```
A failed `POST_RESET` / a previously-hung kernel shows up next run as
`RuntimeError: NOC0 is hung on PCIe device ID N` at mesh-open — that's stale device state, reset again.
(cglagovich's taxonomy: mesh-open / dead-eth-link failure = **machine flakiness**; a hang *during kernel
execution* = **software bug**.)

### Reproduce the hang (bench)
```bash
WAN_GALAXY_LINKS=4 LTX_BENCH_METHODS=fused LTX_BENCH_ONLY=tp2_a_selfattn_qk \
  pytest models/tt_dit/tests/models/ltx/test_ltx_distributed_rmsnorm_bench_fused.py::test_ltx_rmsnorm_bench_galaxy \
  -k tp2 -q
```
- `LTX_BENCH_ONLY=<config_id>` runs one config; `LTX_BENCH_METHODS=fused` skips the composite baseline.
- The bench **catches** the dispatch-timeout exception and reports `RuntimeError` in its table, so the
  pytest "passes" — that's the hang. Set `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` to fail fast instead
  of hanging forever.

### TP=1 correctness control (should PASS)
```bash
pytest models/tt_dit/tests/models/wan2_2/test_wan_fused_distributed_rmsnorm_device_op.py \
  -k tp1_rope -q
```

---

## 4. Debug harness (how we got the data)

`ttexalens` is installed (`uv pip install -r tools/triage/requirements.txt`). Two tools:

### (a) tt-triage — device-state dump on hang
`scripts/run_safe_pytest.sh` wires `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` to tt-triage. BUT the
LTX bench catches the timeout, so run_safe_pytest cleans up the triage log before you read it. Workaround:
run pytest **directly** (not via run_safe_pytest) and inspect the held-hung device manually.

### (b) Watcher + ring buffer — the workhorse (survives a hang; DPRINT does NOT)
DPRINT/DEVICE_PRINT readback is wedged together with the dispatcher on a hang, so it produces nothing.
The **watcher ring buffer** is read out-of-band and works. Recipe:

1. Add `#include "api/debug/ring_buffer.h"` to the kernel and `WATCHER_RING_BUFFER_PUSH(<u32>)` markers.
   (Compute kernels: the push runs on all 3 TRISCs; encode distinct values per location. Watcher must be on.)
2. Run **in the background** with no op-timeout + watcher on (shell `&` was unreliable here; use the
   harness background mode):
   ```bash
   unset TT_METAL_OPERATION_TIMEOUT_SECONDS
   export TT_METAL_WATCHER=10            # poll interval; high enough not to break fabric init
   WAN_GALAXY_LINKS=4 LTX_BENCH_METHODS=fused LTX_BENCH_ONLY=tp2_a_selfattn_qk \
     pytest models/tt_dit/tests/models/ltx/test_ltx_distributed_rmsnorm_bench_fused.py::test_ltx_rmsnorm_bench_galaxy -k tp2 -q
   ```
   (NB: `run_safe_pytest.sh --dev` enables the watcher too but its overhead pushes fabric init past the
   10 s router-sync timeout → use the `TT_METAL_WATCHER` env var directly instead.)
3. Wait ~2–3 min (cold compile + dispatch + hang), then read the latest dump:
   ```bash
   ln=$(grep -n "Device 24 worker core(x= 0,y= 0).*h_id:  9" generated/watcher/watcher.log | tail -1 | cut -d: -f1)
   sed -n "${ln},$((ln+6))p" generated/watcher/watcher.log
   ```
   The `debug_ring_buffer=[...]` line under the worker is **newest-first**. The last marker reached
   pinpoints the wedge.

### (c) ttexalens live PC / debug-bus (no halt needed)
```python
from ttexalens import tt_exalens_lib as lib
from ttexalens.coordinate import OnChipCoordinate
ctx = lib.init_ttexalens(); dev = ctx.devices[24]
loc = OnChipCoordinate(1, 1, "noc0", dev)        # == logical tensix (0,0) == the op worker
db  = dev.get_block(loc).get_debug_bus()
for r in ["brisc","ncrisc","trisc0","trisc1","trisc2"]:
    print(r, hex(db.read_signal(f"{r}_pc")))     # sample twice to see STUCK vs MOVING
```
Kernel ELFs: `~/.cache/tt-metal-cache/5308455294701765855/kernels/{wan_rmsnorm_fused_writer_mux,
wan_rmsnorm_fused_reader,wan_rmsnorm_fused_compute}/<hash>/<risc>/<risc>.elf`. (addr2line symbolization
of runtime PCs fails — kernel runs at a VMA the cache ELF doesn't map; the ring buffer is more reliable.)

---

## 5. Working-tree state — MUST handle before commit / transfer

The only modified file is the compute kernel, carrying **uncommitted debug breadcrumbs**:
```
ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/compute/wan_rmsnorm_fused_compute.cpp
```
- line 42: `#include "api/debug/ring_buffer.h"`
- lines 385, 389, 392, 394, 396: P_NMUL split-init markers (`0xC0130000/0xC0130500/0xC0131000/0xC0132000/0xC0138000`)
- lines 433–451: P_NMUL inner-loop markers (`0xC041xx … 0xC046xx`)

To carry to another host, transfer the patch:
```bash
git diff > /tmp/wan_rmsnorm_breadcrumbs.patch     # then `git apply` on the other host
```
To get a clean tree:
```bash
git checkout -- ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/compute/wan_rmsnorm_fused_compute.cpp
```
The breadcrumbs are diagnostic only — **revert before any real commit.**

---

## 6. Next steps (in priority order)

The hang is `pack_reconfig_data_format(mul_rms_result_cb)` (PACK / `llk_pack_reconfig_data_format`) right
after the matmul-based reduce, only at multi-chunk × ring_size=2. A pure config write shouldn't block, so
the suspect is a **packer "wait-for-idle" / dest-sync that never clears** because the matmul-reduce left
the packer (or the math↔pack dest semaphore) in a state the reconfig waits on. Candidate investigations/fixes:

1. **Try the 2-arg `pack_reconfig_data_format(old_cb, new_cb)`** (`pack.h:167`) with
   `old_cb = reduce_result_cb`, `new_cb = mul_rms_result_cb`. The 1-arg version uses a cached "current
   format"; after a *matmul* reduce that cache may be wrong, and the 2-arg form skips the reconfig when
   formats match. Cheap to try.
2. **Add an explicit packer flush / `reduce_uninit` after the matmul-reduce**, before the P_NMUL reconfig
   — i.e., fix `reduce<>` to call `reduce_uninit` even when `use_matmul` (currently skipped at
   `reduce_helpers_compute.inl:535`), or call the matmul/packer teardown in the compute kernel. Rationale:
   `UPMD` (matmul-unpack) is live on the unpacker; the reduce's matmul path may not be torn down, and the
   packer reconfig waits on it. (Test even though TP=1 also skips it — the multi-chunk context may be what
   makes the un-torn-down state fatal.)
3. **Confirm which thread is truly stuck** with `MATH(( WATCHER_RING_BUFFER_PUSH(...) ))` /
   `PACK(( ... ))` / `UNPACK(( ... ))`-gated markers around the 3 init calls (disambiguates the
   shared-ring-buffer multi-thread noise; current markers are ungated).
4. **Compare with the working PRE pack_reconfig** (`wan_rmsnorm_fused_compute.cpp:218`,
   `pack_reconfig_data_format(pre_intermediate_cb)`) which runs fine — diff the packer/dest state that
   precedes it vs the POST one (post-reduce vs post-PRE-sum-of-squares).
5. If it proves to be a genuine LLK/HW packer-reconfig-after-matmul-reduce issue, the pragmatic op-level
   fix is to **avoid the matmul reduce → bcast-mul transition** at TP>1 — e.g., pre-sum the `ring_size`
   gathered stat tiles with an eltwise add so the reduce is single-tile, or restructure POST so the
   packer is flushed/re-init'd cleanly between the reduce and the norm-mul.

### Interim safety option
The shipped `chunk_size_rows=1` clamp is correct for the **TP=4 primary target + Wan** (those don't hang).
The open hang is **TP=2 per-head-rope configs that fit in L1**. If a real fix is deferred, add a fail-fast
`TT_FATAL` for the `ring_size==2 && multi-chunk && per_head_rope` combination so it errors cleanly instead
of wedging the device (and needing the flaky reset).

---

## 7. Pointers
- Bench suite: `models/tt_dit/tests/models/ltx/test_ltx_distributed_rmsnorm_bench_fused.py`
- Findings/coverage: `models/tt_dit/tests/models/ltx/LTX_RMSNORM_FINDINGS.md`
- Compute kernel: `ttnn/.../wan_fused_distributed_rmsnorm/device/kernels/compute/wan_rmsnorm_fused_compute.cpp`
- Reduce helper (CB semantics audited here): `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.{hpp,inl}`
- MUX writer: `ttnn/.../device/kernels/dataflow/wan_rmsnorm_fused_writer_mux.cpp`
- Reader: `ttnn/.../device/kernels/dataflow/wan_rmsnorm_fused_reader.cpp`
