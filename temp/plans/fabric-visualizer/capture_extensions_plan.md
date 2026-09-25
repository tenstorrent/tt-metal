# Task 3 (rest) — raw L1 image, liveness sequence, identity (Python capture)

Parent: `visualizer_plan.md` §"3. Capture: raw image, liveness, identity". First cut is done and
described in `capture_plan.md` (manifest load, router enumeration, coordinate check, reset + wall
clock, streams 0–29, snapshot schema, CLI, tests, 40/40 routers on T3K). Task 2 (`region_tree_plan.md`)
is done, so the manifest now names every region and every allocated stream id per router.

Goal: after this task, a capture contains **every byte decode will ever need**, so that no hang has to
be re-captured once `decode/` exists. Capture still interprets nothing.

---

## 1. What we learned reading the code (drives every decision below)

### 1.1 ttexalens surface (installed: `/opt/venv/lib/python3.10/site-packages/ttexalens`, `tt-exalens` 0.3.32)

| Need | Reality |
|---|---|
| Bulk read | `read_from_device(location, addr, device_id=0, num_bytes=4, context=None, noc_id=None, safe_mode=None) -> bytes`. Returns `bytes`. No documented size cap, **no burst/chunking helper** — one call is one read; UMD handles unaligned head/tail and picks DMA over NOC when `len >= context.dma_read_threshold` (24) on MMIO non-BH chips. |
| Word reads | `read_words_from_device(..., word_count) -> list[int]`, little-endian. |
| ASIC join | `device.unique_id: int` (set from UMD topology discovery), `context.device_by_unique_id: dict[int, Device]`, `context.cluster_descriptor.get_chip_unique_ids() -> dict[chip_id, unique_id]`. `context.devices` is keyed by **cluster chip id**, which is what we index today. |
| Version | No `ttexalens.__version__`. `importlib.metadata.version("tt-exalens")` → `"0.3.32"`; `"tt-umd"` → `"0.9.9"`. |
| Reset | `RISCV_DEBUG_REG_SOFT_RESET_0` is offset `0x1B0` off debug base `0xFFB12000` on **both** WH and BH eth blocks, so the dumper's hardcoded `0xFFB121B0` is correct on BH too. The **bit** differs: WH `erisc` `reset_flag_shift=11`; BH `erisc0`=11, `erisc1`=12. |
| Eth L1 size | WH 256 KiB, BH 512 KiB. |

There is no `server_ifc` on `Context` (it is `umd_api` / `file_api`), and there is no `"virtual"`
coordinate system — Metal's "virtual" is ttexalens `"translated"`. Neither is needed here: we keep
peeking with the `OnChipCoordinate` from `get_block_locations("eth")[eth_chan]`.

`context.find_device_by_id()` accepts a chip id **or** a unique id. Do not use it for the ASIC join —
a unique id that happens to be a small integer would silently resolve as a chip id. Use
`device_by_unique_id` and fail loudly.

### 1.2 The manifest's own numbers (live T3K WH, FABRIC_2D)

```
hal.unreserved        base 98304   size 154624   end 252928      <- the image
hal.fabric_telemetry  base 258976  size 160
hal.routing_table     base 259136  size 2704     <- routing_l1_info_t
hal.router_state      base 259136  size 4        <- first word of the same struct
hal.router_command    base 259152  size 4
hal.go_msg            base 254432  size 36       <- ABOVE unreserved end
hal.launch            base 253280  size 144      <- ABOVE unreserved end
heartbeat.address     8064 (0x1F80)              <- BELOW unreserved base
router_template.edm_status_address        98608  <- inside the image
router_template.termination_signal_address 98576 <- inside the image
```

Two consequences the earlier plan text did not spell out:

- **The heartbeat and `go_msg` are not in the image.** `0x1F80` is below `UNRESERVED` and `go_msg` /
  `launch` are above it. They are separate small reads, not slices. (BH heartbeat is `0x7CC70`, also
  outside.)
- **`EDMStatus` and the termination word *are* in the image.** Liveness still reads them explicitly as
  4-byte reads because classification happens before the 151 KiB read; the image then contains a second
  copy. A disagreement between the two is real evidence of a moving router, so record both rather than
  deduplicating.

Sizes: 154624 B image × 40 routers ≈ **5.9 MB** per T3K snapshot, plus ~2.9 KB of fixed siblings per
router. That does not belong in JSON — hence the `.bin` sidecar the parent plan already decided on.

### 1.3 Structures capture must locate (not decode)

- `routing_l1_info_t` (`fabric_common.h:867`): `RouterStateManager state_manager` (32 B), then
  `uint16_t my_mesh_id` at **+32**, `uint16_t my_device_id` at **+34**. These two are the device-pairing
  check from the parent plan (no `launch_id`). Capture surfaces them as scalars *and* keeps the whole
  2704 B region raw.
- `go_msg_t` (`dev_msgs.h:222`) is a packed union of one `uint32_t`; `signal` is the **high byte**
  (`dispatch_message_offset, master_x, master_y, signal`). The HAL `go_msg` region is
  `go_messages[go_message_num_entries]` (36 B here), so capture reads the region raw and records word 0;
  decode picks the entry.

---

## 2. Decisions

1. **`.bin` sidecar, one per snapshot.** Path defaults to the output path with suffix `.bin`
   (`fabric_debug_snapshot_rank_1_of_1.json` → `...json.bin`? no — `...snapshot_rank_1_of_1.bin`,
   i.e. `output.with_suffix(".bin")`). Written first, then the JSON that references it. Both atomic
   (`tmp` + `rename`), JSON last so a JSON on disk always has its bytes.
2. **Everything raw is a blob.** Not just the image: the telemetry region, routing table, launch, and
   `go_msg` are blobs too. A blob entry is
   `{ name, address, size, offset, sha256, status }` and the bytes live in the `.bin`. JSON keeps only
   small scalars. One uniform mechanism, and a new HAL region later costs one line.
3. **Streams come from the layout, not from `range(30)`.** Task 2 made the manifest authoritative;
   §"validation" of this branch already proved the dumper's 0–29 grouping is wrong for this fabric.
   Peek every **allocated** `stream_reg` id in the router's layout (a superset of enabled — an
   allocated-but-disabled stream reading non-zero is evidence). `--streams all` stays as an escape
   hatch that reads 0–31 for comparing against the old dumper.
4. **Pre/post stream bracket.** Streams are read before and after the image. Any id whose value moved
   sets `streams.torn = true` and the router's status to `torn`. Never hide the difference: store both
   maps.
5. **Status taxonomy replaces `ok`.** `status: ok | unreadable | reset | torn | unknown | unsupported`
   on the router, and the same enum per blob. `ok: bool` is **removed**, not kept alongside —
   `status == "ok"` is the same bit and two sources of truth rot. Nothing outside this branch consumes
   the snapshot, so `snapshot_version` stays **1**, same call we made for the manifest.
6. **Liveness rounds come first, then the image pass.** N rounds (default 3) at T seconds (default 1.0)
   read only heartbeat + wall clock for *every* router, then one heavy pass does status words, image,
   siblings, and the stream bracket. Rationale: heartbeat classification needs samples that are close
   together **across the fleet**; a 6 MB image pass takes seconds and would smear the cadence if
   interleaved. Every round and the image pass carry their own timestamps so decode can see the gap.
7. **Chunked reads from day one.** `read_from_device` has no documented cap and no internal chunking,
   and a 151 KiB single read over a wedged NOC is exactly where this would break. Read in
   `--read-chunk` (default 64 KiB) pieces. A failed chunk marks the blob `unreadable` and zero-fills
   that span (recorded in the blob's `error`), so offsets stay stable and the rest survives.
8. **ASIC join, with `physical_chip_id` as a cross-check.** Build `unique_id -> device` from
   `context.device_by_unique_id`. Join on `int(chip.asic_id, 16)`. If it resolves to a different device
   than `context.devices[physical_chip_id]`, trust the ASIC id and record
   `asic_id_matches_physical: false`. If the manifest's `asic_id` is null, fall back to the chip id with
   a warning.
9. **`owner_alive` is best effort and never blocks.** Scan `/proc/*/fd` for links into
   `/dev/tenstorrent/`, excluding this pid. Permission errors → `null` (unknown), not `false`.

---

## 3. Snapshot shape (v1, changed in place)

```json
{
  "snapshot_version": 1,
  "kind": "fabric_debug_snapshot",
  "captured_at": "...",
  "manifest": { "path": "...", "manifest_version": 1, "sha256": "...", "run": { ... } },
  "provenance": {
    "ttexalens_version": "0.3.32",
    "tt_umd_version": "0.9.9",
    "hostname": "...",
    "owner_alive": true,
    "argv": ["..."]
  },
  "raw": { "file": "fabric_debug_snapshot_rank_1_of_1.bin", "size": 6116480, "sha256": "..." },
  "samples": [
    {
      "capture_time": "...",
      "routers": [
        {
          "id": { "mesh_id": 0, "chip_id": 0, "eth_chan": 0 },
          "physical_chip_id": 4,
          "asic_id": "0x000000035172312b",
          "asic_id_matches_physical": true,
          "status": "ok",
          "error": null,
          "health": { "reset": 0, "reset_bits": { "erisc0": false, "erisc1": null }, "wall_clock": 123 },
          "lifecycle": { "edm_status": 2746467283, "termination_signal": 0, "go_signal": 0 },
          "identity": { "my_mesh_id": 0, "my_device_id": 0, "matches_manifest": true },
          "liveness": [ { "t": "...", "heartbeat": 3703177280, "wall_clock": 1 } ],
          "streams": {
            "pre":  { "0": { "buf_space_available": 8 } },
            "post": { "0": { "buf_space_available": 8 } },
            "torn": false
          },
          "blobs": {
            "unreserved":       { "address": 98304,  "size": 154624, "offset": 0,      "sha256": "...", "status": "ok" },
            "fabric_telemetry": { "address": 258976, "size": 160,    "offset": 154624, "sha256": "...", "status": "ok" },
            "routing_table":    { "address": 259136, "size": 2704,   "offset": 154784, "sha256": "...", "status": "ok" },
            "go_msg":           { "address": 254432, "size": 36,     "offset": 157488, "sha256": "...", "status": "ok" },
            "launch":           { "address": 253280, "size": 144,    "offset": 157524, "sha256": "...", "status": "ok" }
          }
        }
      ]
    }
  ]
}
```

Schema rules: `status` is an enum (`ok|unreadable|reset|torn|unknown|unsupported`); `blobs` is an open
object (`additionalProperties: $defs/blob`) keyed by HAL region name so a new region needs no schema
edit; `streams.pre`/`post` keep the current numeric-string keys but widen the pattern to `0–31`
(BH pins VC2 at 30/31); `raw` is required, `liveness` has `minItems: 1`.

---

## 4. Implementation, module by module

### `manifest.py`
- `RouterTarget` gains `asic_id: int | None` (parsed from the chip's hex string) — the chip loop
  already has the chip object in hand.
- `stream_regs_for_router(..., enabled_only: bool = False)`. Today's helper returns enabled-only;
  capture wants allocated. Default flips to allocated, and the one existing test asserting `(22,)` for
  the fixture becomes two assertions (allocated `(22, 23)`, enabled `(22,)`).
- `hal_regions()` → the ordered list of `(name, base, size)` blobs to read: `unreserved`,
  `fabric_telemetry`, `routing_table`, `go_msg`, `launch`. Skip any region with `size == 0`
  (`eth_fw_mailbox` is `0/0` on WH — that is `unsupported`, not an error).

### `peek.py`
Split the current monolith:
- `read_u32(device, loc, address)` (unchanged behaviour) and new `read_block(device, loc, address, size, chunk)` → `(bytes, status, error)`.
- `resolve_devices(manifest, context)` → `{(mesh, chip): device}` via the ASIC join, plus the
  physical-id cross-check. This replaces `_lookup_device`.
- `liveness_round(targets, resolved, read_u32, heartbeat_address)` → per-router `{t, heartbeat, wall_clock}`.
- `peek_router(...)` becomes the ordered sequence: coordinate check → reset/wall clock → pre-streams →
  `edm_status` / termination / `go_msg` word → blobs → post-streams → status roll-up.
- Status roll-up rules, in priority order: any read raised → `unreadable`; reset bit set for this
  arch's erisc → `reset`; pre ≠ post → `torn`; otherwise `ok`. `unknown` is for a router we never got
  to (no device); `unsupported` is a zero-size HAL region.

### `rawfile.py` (new)
```python
class RawBlobWriter:
    def __init__(self, path): ...
    def add(self, payload: bytes) -> tuple[int, int, str]:  # offset, size, sha256
    def close(self) -> tuple[int, str]:                     # total size, file sha256
```
Buffers to a `tmp` file next to the target and renames on `close()`. Keeps a running `hashlib.sha256`
for the whole file and a per-blob digest. A thin, separately testable seam so the snapshot builder
never touches file handles.

### `snapshot.py`
- `build_snapshot(manifest, samples, raw_reference, provenance, captured_at=None)`.
- New `provenance()` helper (`importlib.metadata`, `socket.gethostname`, `owner_alive()`, `sys.argv`).
- `owner_alive()` lives here, not in `peek.py`: it is host provenance, not a device read.

### `cli.py`
- New flags: `--liveness-samples N` (default 3), `--liveness-interval T` (default 1.0),
  `--streams {layout,all}` (default `layout`), `--read-chunk BYTES` (default 65536),
  `--raw PATH` (default `output.with_suffix(".bin")`), `--no-l1-image` (skip the `unreserved` blob only;
  everything else still runs, for a fast triage capture).
- `make_read_u32` stays; add `make_read_block` wrapping `read_from_device` with the same
  exception-to-`None` discipline.
- Summary line becomes a status histogram (`40 routers: 38 ok, 1 torn, 1 reset`) instead of a bool count.

---

## 5. Tests

**Offline** (`capture/tests/`, no devices, `--noconftest`):
1. `RawBlobWriter` round-trip: three blobs → offsets are contiguous, per-blob `sha256` matches the
   slice read back out of the file, file digest matches the whole file.
2. Blob offsets in the snapshot actually address the right bytes: fake reader returns a recognisable
   pattern per region; test re-reads the `.bin` by `offset/size` and asserts the pattern.
3. Torn detection: fake reader returns a different stream value on the post pass → `streams.torn`,
   router `status == "torn"`.
4. Status precedence: unreadable beats reset beats torn.
5. ASIC join: two fake devices, manifest `asic_id` pointing at the one whose `physical_chip_id`
   disagrees → the ASIC-chosen device is used and `asic_id_matches_physical` is false.
6. Liveness: `--liveness-samples 3` produces three entries per router with distinct `t`, and the
   sleep is injected (no real 3 s in unit tests).
7. Layout-driven stream selection: fixture layout allocates ids `{22, 23}`, only `22` enabled → both
   are peeked; `--streams all` peeks 0–31.
8. Chunking: `--read-chunk 16` against a 40-byte region issues three reads and reassembles exactly;
   a failing middle chunk marks the blob `unreadable` and leaves the neighbours intact.
9. Schema validation of the produced snapshot (extends the existing test).

**Hardware, T3K** (manual, recorded in the README):
1. 40 routers, all `status: ok`, every `unreserved` blob exactly 154624 bytes, `.bin` ≈ 6.1 MB.
2. `lifecycle.edm_status` decodes to `READY_FOR_TRAFFIC` (2746467283) on an idle fabric.
3. `identity.my_mesh_id/my_device_id` equals the manifest row for all 40.
4. Heartbeat advances across the three rounds on an idle fabric (this is the first time we prove the
   kernel's heartbeat actually moves).
5. **Kill-then-capture retention check**: `SIGKILL` the fabric owner without `close_device`, capture
   again, confirm the image is neither zeroed nor reset and the heartbeat is now static. This is the
   check the whole tool's premise rests on; write the result into the README either way.
6. Wall-clock cost of the image pass, measured. If it is minutes rather than seconds, `--no-l1-image`
   becomes the documented default for a first look.

**Blackhole**: no new CI entry (capture is not in any pipeline). Manual run on a BH box if one is
free, checking the two BH-specific things: reset bits 11 **and** 12 are reported per erisc, and the
heartbeat address is `0x7CC70` from the manifest rather than the WH constant.

---

## 6. Slices (each builds, each reviewable on its own)

| # | Slice | Files | Done when |
|---|---|---|---|
| A | ASIC join + provenance | `manifest.py`, `peek.py`, `snapshot.py`, `cli.py`, schema | snapshot carries `asic_id`, `provenance`; existing tests pass with the join in place |
| B | Status taxonomy + liveness rounds | `peek.py`, `snapshot.py`, `cli.py`, schema, tests | `ok` gone, `status` + `liveness[]` in, unit tests 3–6 green |
| C | Raw blobs + `.bin` | new `rawfile.py`, `peek.py`, `snapshot.py`, `cli.py`, schema, tests | unit tests 1, 2, 8 green; T3K writes a 6 MB sidecar |
| D | Layout-driven pre/post streams | `manifest.py`, `peek.py`, tests | unit test 7 green; blind 0–29 gone |
| E | Hardware validation | README | T3K list above recorded, kill-then-capture answered |

Stop for review after **B**, after **C**, and after **E**.

---

## 7. Risks and how the plan handles them

- **A 151 KiB read is the one unproven ttexalens call.** Chunked from the start (decision 7), chunk
  size is a flag, and a failed chunk degrades one blob instead of the capture.
- **Capture duration.** 40 routers × 151 KiB over NOC, unknown throughput. Measured in slice E;
  `--no-l1-image` is the escape hatch, and the liveness rounds run *before* the image so a slow image
  pass never invalidates the cadence.
- **`asic_id` absent or stale.** Manifest rows written before task 1's `asic_id` landed, or a manifest
  from a different boot. Fall back to `physical_chip_id` with a warning; the identity words read out of
  `routing_l1_info_t` are the independent check and they are per-router, not per-chip.
- **BH 2-ERISC reset.** One `SOFT_RESET_0` word holds both bits (11, 12). Capture stores the raw word
  and the decoded per-erisc booleans; on WH `erisc1` is `null`, not `false`.
- **Heartbeat semantics.** `0xDCBA0000 | counter` — a router that never started writes something that
  is not in that format at all. Capture stores raw; classification (`advancing` / `static` /
  `not_fabric_format`) is decode's job, per the parent plan's split.
- **Snapshot shape change with tests already green.** `ok` is removed in the same commit as its
  replacement and `test_peek.py` / `test_cli.py` are updated in that commit; no compatibility shim.
- **Multi-host.** Still untestable on this single-rank T3K. The ASIC join is the piece that matters
  there and it is unit-tested with fakes.
- **`safe_mode`.** Default `True` in ttexalens. Eth L1 is a safe region, so the image read is fine; if
  a future region trips it, surface the exception as `unsupported` rather than silently passing
  `safe_mode=False`.

---

## 8. Open questions (not blocking)

- Should `--no-l1-image` be the default if the image pass turns out to be slow? Decide with the
  measurement in slice E, not now.
- The `.bin` is per snapshot; a Galaxy capture would be ~10× larger. Per-router files or compression
  can be added later without a schema change (the blob entry already names its file).
- `owner_alive` via `/proc` needs permission to stat other users' fds on a shared box. Unknown is a
  legitimate answer; we are not adding a privileged helper.
- Whether decode wants the `launch` region at all. It is 144 B, so it is cheaper to capture it now than
  to re-capture a hang later.
