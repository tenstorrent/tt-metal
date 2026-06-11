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

---

## 8. Update (chunk=1 relaxation attempt) — matmul-reduce layer FIXED; two layers remain

Goal: relax the per-head-RoPE `chunk_size_rows=1` clamp to recover AG amortization +
read/compute overlap on the big TP=4 shapes (`selfattn_qk_s2` is 308us, only ~52% compute).

**Layer 1 (the §2 wedge) — FIXED** by commit `11bc6a0e056`: the POST `reduce<AVG,REDUCE_ROW>`
was a tile **matmul** (`reduce_uses_matmul<AVG,ROW>==true`); replacing it with an explicit
FPU eltwise-add of the `ring_size` gathered partial-sum tiles (+`*1/H_full+eps+rsqrt`)
removes the matmul→pack transition the packer reconfig wedged on. Verified: the §1 repro no
longer wedges *eagerly*, clamped TP=4 is PCC 99.9991% + perf-neutral, and the 2-arg
`pack_reconfig` (§6.1) did NOT help (so it was the matmul, not the reconfig args).

**Layer 2 — a chunk>1 reader NoC-read hang (NOT yet fixed).** With the clamp disabled
(`WAN_RMSNORM_NO_PERHEAD_CLAMP=1`), chunk>1:
- `a2v_videoQ_s2` (feat=512 → num_tile_cols=16 → chunk=8, 19 chunks): **runs, correct, ~13%
  faster** (traced 208.6→182.1us). The relaxation pays off here.
- `selfattn_qk_s2` (feat=1024 → num_tile_cols=32 → chunk=4, 38 chunks): **hangs** under
  sustained *traced* execution (eager runs fine). Heisenbug: **vanishes under TT_METAL_WATCHER**
  (polling perturbs timing), so the §4(b) ring-buffer recipe can't catch it.
- tt-triage on the live no-watcher wedge (run pytest directly, no op-timeout, then
  `python3 tools/tt-triage.py`): op 10 RUNNING; `check_noc_status` shows the **reader (brisc)
  on the op worker cores with ~2300 NoC reads issued but unreturned**
  (`NIU_MST_RD_RESP_RECEIVED`: issued 16707 vs recv 14397). Op kernels don't symbolize in
  `dump_callstacks` (addr2line/VMA issue, as in §4c). So: reader-side NoC read non-completion,
  correlated with MORE chunks (38 vs 19) and/or num_tile_cols=32 (cos/sin = 2*32=64 reads per
  per-row barrier vs 32 for a2v).

**Layer 3 — partial-last-chunk correctness (NOT yet fixed).** chunk>1 shapes whose tile-row
count isn't divisible by the chunk (e.g. `*_s1` = 1216 rows = 38 tile-rows, chunk=4 → 2-row
tail) give **94% PCC** (vs 99.99% for evenly-divisible 152/8). The clamp (chunk=1) hid this.

Next-step suspects for layer 2: reader outstanding-read management for chunk>1 (cap cos/sin
reads per barrier like `input_barrier_tiles` does for input?); CB depth/wrap for input_cb /
cos-sin CB under many chunks; interaction of the deep-read reader with the AG mcast on the NoC.
The heisenbug nature (watcher-masked) means try targeted reader/CB changes + the no-watcher
run_safe_pytest repro (`-x`, 5s dispatch timeout) as the pass/fail oracle, not the watcher.

---

## 9. Update (layer-3 root-cause) — it's a PRE-EXISTING, GENERAL "last row of each chunk" bug

Investigated layer-3 (the chunk>1 correctness bug) with fast eager per-tile-row diffs
(fused vs composite+standalone baseline). Findings:

- **Not the partial last chunk — it's the LAST ROW of EVERY chunk.** The error is periodic
  with period = chunk_size_rows. On `selfattn_qk_s1` (1216 rows=38 tile-rows; 13 workers ×
  ~3 rows ⇒ chunk_size_rows=3, 1 chunk/worker), tile-rows 2,5,8,…,35 (each worker's LAST
  row) are wrong by ~9-27; all other rows are bf16 noise (~0.1).
- **General — not rope- or head-specific.** Reproduces identically on `tp4_v_block_s1`
  (num_heads_per_device=1, NO rope, adaLN bias), `tp4_v_textcross_q_s1` (8 heads, NO rope),
  and the rope QK shapes. So it's neither the per-head RoPE nor the head-split path.
- **PRE-EXISTING — not candidate `11bc6a0e056` (eltwise reduce).** Reverting the compute
  kernel to the pre-#3 matmul reduce gives the byte-identical wrong pattern. So the
  eltwise-reduce commit is exonerated.
- **It's GARBAGE DATA, not a wrong rms scale.** On a bad row, fused/baseline ratio has
  mean≈1.54, std≈0.51, range −0.83→4.52 (sign flips); the adjacent good row is 1.0000
  (std 0.0025). A wrong per-row rms would be a *uniform* scale — this is varied per element,
  so the last row's INPUT or OUTPUT data is wrong, not its normalization factor.
- **Shipping implication.** `block` and `text-cross QK` norms are NOT per-head-rope-clamped,
  so they ALREADY run chunk>1 in production with this bug — i.e. multi-chunk block/text-cross
  RMSNorm has been silently producing a wrong last-row-per-worker. (per-head-rope QK norms
  were saved by the chunk=1 clamp.)

Remaining gap: pin the exact buffer. Leading suspects (all "last row of the chunk" handling):
the reader's per-row input deep-read / input_cb residency for the final row; the writer's
per-row output drain (`noc_async_writes_flushed` + per-row pop, final barrier only at kernel
end); or the AG packed-page scatter tail. Next step: dump the compute's input_cb and output_cb
for the last row of a chunk (kernel ring-buffer or a DRAM scratch copy) on a tiny chunk=2
repro, comparing last-row vs first-row bytes. This is eager-testable (fast, safe, no hang).

---

## 10. SOLVED — root cause is a multi-hop fabric-multicast farthest-target drop; fix = clamp chunk=1 for >2-hop AG

Root-caused the chunk>1 correctness bug (sec 9) to the **fabric multicast**, not op logic:

- The bad row's output is a UNIFORM **2.0x** scale (no-bias textcross: ratio mean 2.0055,
  std 0.034) ⇒ 1/rms is 2x too large ⇒ mean(x²) 4x too small ⇒ with ring_size=4 the gathered
  sum holds only the LOCAL partial; the 3 remote partials are missing.
- Direct DRAM-scratch (`pob`) dump for chunk 0 (chunk_size_rows=2): device-1 (1 hop) and
  device-2 (2 hops) pages have BOTH rows (~32000); **device-3 (3 hops) has row 0 (~32000) but
  row 1 = 25.6 (stale)**. The FARTHEST multicast target loses everything past the first row.
- `flush=true` on the fused write+atomic_inc: NO effect. Per-row single-flit packets: NO
  effect (device-3's 2nd packet still drops; its inc still arrives → no hang). So it is a
  fabric/EDM multi-hop multicast delivery bug, independent of op packet structuring.
- Hop-dependent: TP=2 (1 hop) and chunk=1 (single-row page, no tail) are always correct —
  which is exactly why TP=2 chunk>1 (Wan) and all chunk=1 paths worked.

**Pre-existing production bug:** `block` and `text-cross QK` norms are NOT per-head-rope
clamped, so they ran chunk>1 at TP=4 and were SILENTLY WRONG on each chunk's last row.

**Fix (shipped here):** force `chunk_size_rows=1` whenever the AG path spans >2 hops
(`use_mux && ring_size > 2`), in BOTH `compute_sizing` and the program factory. Verified:
all 8 representative TP=4 configs (block s1/s2, text-cross q s1/s2, self-attn qk, a2v video-Q,
a2v audio-K) now PCC 99.998-100.000% vs baseline; full TP=4 bench runs with no hang.

**Still open (the original perf goal):** the chunk>1 RELAXATION is fabric-blocked at TP=4 —
recovering it needs a fabric-team fix to the multi-hop multicast (deliver the full payload to
the farthest target) or an AG redesign that avoids multi-row multicast (e.g. per-device
point-to-point with flow control). At TP=4 the op is now correct but pinned to chunk=1.

---

## 11. ACTUAL ROOT CAUSE (supersedes §9–10) — buffer/kernel chunk_size mismatch from num_links

§9–10's "fabric multicast farthest-target tail-drop" was WRONG (the `pob` dump on a
38-row shape was a red herror). The real bug is a **sizing inconsistency** between the
DRAM stats buffer and the kernel — nothing to do with the fabric, and the clamp is NOT
the fix.

Decisive evidence: with chunk>1 (NO_PERHEAD_CLAMP), `selfattn_qk_s1` (38 tile-rows)
fails at 94% but `selfattn_qk_s2` (152 tile-rows) passes at 99.9997% — BOTH chunk=3,
both per-head ROPE, both multi-hop. A fabric tail-drop would hit both. The rowdiff
showed the bad rows are exactly the **last row of every chunk** at a uniform **2x**
scale (mean(x²) 4x too small ⇒ only the local partial summed). The `pob` dump showed
the buffer was `[1,1,76,64]` = **window=2, ncd=19 (chunk=2)** while the kernel CT args
were **chunk=3, ncd=13 (window=3)**.

Why they disagree:
- `compute_sizing` (sizes the buffer via `create_stats_buffer`) picked `num_workers=19`
  for 38 rows ⇒ rows_per_worker=2 ⇒ **chunk=2**, and did NOT apply the num_links rounding.
- `create_at` (the actual program) rounds `num_workers` DOWN to a multiple of
  `num_links_eff=4` ⇒ `16` ⇒ rows_per_worker=3 ⇒ **chunk=3**.
- Separately, `create_stats_buffer` hardcoded `num_links=1`, while the op is invoked with
  `num_links=4`. The buffer geometry depends on num_links (through the rounding), so the
  two were computed for different link counts.

The writer then emits 3-row (96 B) pages into a buffer laid out for 2-row (64 B) pages
⇒ the scatter reads garbage into each chunk's last AG row. Shapes where `pick()` was
already a multiple of num_links (152 → 64 workers, no rounding) happened to agree, which
is exactly why only *some* shapes were corrupted, and only at num_links>1.

**Fix (shipped, no clamp):**
1. `compute_sizing`: apply the same `num_links` rounding to `num_workers` that
   `create_at` does, so buffer geometry == kernel geometry.
2. `wan_fused_distributed_rmsnorm_create_stats_buffer`: add a `num_links` parameter
   (was hardcoded to 1); callers pass the SAME value used to invoke the op.
3. LTX bench + `models/tt_dit/layers/normalization.py`: forward `num_links` (bench) and
   weight/RoPE (both) into `create_stats_buffer` so its chunk/clamp matches the op.

**Verified:** every TP=4 shape with chunk>1 (NO_PERHEAD_CLAMP) now PCC 99.98–100.00%
(worst selfattn_qk_s1 99.982%); default path 99.999%; full timed bench runs with no hang.
The per-head-ROPE clamp remains for its INDEPENDENT reason (cos/sin CB L1 overflow at
wide features), not the AG.
