T3K 2D FABRIC DEBUG-TOOL HANG — EXPERIMENTAL, NOT HARDWARE-VALIDATED

Purpose
-------
A deliberately non-draining fabric receiver with normal FABRIC_2D mesh routing.
This is a synthetic backpressure fault, NOT a natural circular-routing deadlock.
It is intended to leave occupied receive buffers, exhausted upstream credits,
and eventually blocked sender queues available to a live debugger.
It does not spin forever inside the receiver handler or halt the eRISC CPU.

The one-file patch gates receiver forwarding for packets whose payload_size_bytes
is exactly 1008, only when FABRIC_2D is compiled. The router main loop, arrival
ACKs, ordinary space checks, and completion processing remain in place.
An unforwarded packet cannot complete and return its receiver-slot credit.
Other packets can be blocked behind the selected packet (head-of-line blocking).

Scope / caution
---------------
Use a reserved T3K and no unrelated workloads. Packet size is the ONLY traffic
selector: ANY 1008-byte payload in 2D mode, including another workload's packet,
will be blocked while this patch is installed. This is not a production option.
Do not run the earlier experimental clockwise-route patch with this package.
The previous 1D bubble-control change is unnecessary; restore its original
expression manually if you want a clean experiment. This installer does not
change or revert that existing edit, or any other source outside its marked block.

The YAML uses mesh 0's adjacent 2x2 region, specified by mesh coordinates. It does
not assume that physical device IDs or logical chip IDs are row-major. All eight
T3K devices may be opened; four have the explicit test senders and destinations.
Confirm the log reports FABRIC_2D (not FABRIC_1D_RING), successful initialization,
and program launch. A hang before workload launch is NOT a successful result.

Setup / run (from tt-metal root)
-------------------------------
Set KIT to the extracted directory. Keep all package files together.

  KIT=/absolute/path/to/fabric_2d_debug_hang
  python3 "$KIT/patch_receiver.py"            # preview the exact proposed diff
  python3 "$KIT/patch_receiver.py" --apply    # changes one device-kernel file
  bash "$KIT/run.sh" control                 # 992-byte control; should complete
  bash "$KIT/run.sh" hang                    # 1008-byte traffic; intended to stall

Use your already-built test_tt_fabric executable. Only device-kernel source is
modified. run.sh forces device JIT compilation and selects your current checkout.
It disables the operation timeout for the hang run. It intentionally does NOT
pass --show-progress or --show-progress-detail (benchmark counter mismatch).
--wait-on-hang is not needed because that monitor is not being used.

Once the workload stalls after launching, leave its process alive and inspect
from a second terminal with your own debugger. Do not reset the devices until
captures are finished. A finite test/control timeout can abort the host; run it
only as the control, not while trying to preserve an intentional hang.

Expected observations (not hardware-verified)
-------------------------------------------
- Actual 2D receiver slots retain the selected packets.
- The upstream free-receiver-slot credit decreases without replenishment for them.
- Sender queues eventually fill and workers block.
- Heartbeats may continue: executing a loop does not imply packet progress.
- Arrival-notification counters can become zero after first-level ACKs even when
  acknowledged packets remain in receiver slots. Do not equate that with empty RX.
- This fault may stop traffic at its FIRST receiving router, not distribute a
  natural circular wait throughout the mesh. It is for debugger-state testing.

Restoration / stronger control
-----------------------------
After capturing and stopping the run, recover/reset the reserved hardware using
your normal site procedure as needed. Do NOT reset during a capture.

  python3 "$KIT/patch_receiver.py" --undo

--undo removes only the exact inserted block and preserves other edits. It refuses
if the block itself was modified. Force JIT compilation again after undoing.
For an exact same-payload control after removal, run:

  TT_METAL_RUNTIME_ROOT="$PWD" TT_METAL_KERNEL_PATH="$PWD" \
  TT_METAL_FORCE_JIT_COMPILE=1 \
  ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
    --test_config "$KIT/test_fabric_2d_debug_hang.yaml" \
    --filter name.Debug2DHang --master-seed 1 --dump-built-tests --show-workers

If the patched run completes, check that the patch is present in the checkout
used by JIT, and inspect the expanded traffic configuration / actual payload size.
If initialization or the 992-byte control stalls, stop: do not label that the
intended traffic-backpressure reproduction. The selector may have caught other
traffic, the device may need recovery, or the checkout may differ from indexed main.

Validation limits
-----------------
The patch anchors, Python/shell syntax, YAML structure, and apply/remove roundtrip
were checked locally. No TT toolchain build or T3K execution was possible here.
This is a small source-grounded fault-injection experiment, not a claim of a
hardware-proven reproducer or a guarantee that your decoder will display a
particular occupancy label.
