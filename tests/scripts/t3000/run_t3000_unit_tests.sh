#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_t3000_ttmetal_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttmetal_tests"
  ./build/test/tt_metal/distributed/distributed_unit_tests

  echo "LOG_METAL: Testing TT_VISIBLE_DEVICES functionality"
  ./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh ; fail+=$?

  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectRingGatherAllChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueMultiDevice*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleDevice*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQMultiDevice*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter="DPrintMeshFixture.*:MeshWatcherFixture.*" ; fail+=$?

  # Programming examples
  ./build/programming_examples/distributed/distributed_program_dispatch
  ./build/programming_examples/distributed/distributed_buffer_rw
  ./build/programming_examples/distributed/distributed_eltwise_add
  ./build/programming_examples/distributed/distributed_trace_and_events

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttfabric_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttfabric_tests"
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3k*MeshGraphFabric2DDynamicTests*

  # originally were in TT-NN, now promoted to TT-Metal (Fabric)
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*WorkerFabricEdmDatapath*:*EdmFabric*"
  # Instantiate a 1x8 Mesh on a T3K with 2D Fabric
  TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2DFixture.TestUnicast*"

  # TODO (issue: #24335) disabled slow dispatch tests for now, need to re-evaluate if need to add in a different pool
  #TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

  # Offline test for Cluster Validation Tool
  ./build/tools/scaleout/run_cluster_validation --global-descriptor-path tools/tests/scaleout/global_system_descriptors/proto/4_lb_superpod_physical_desc.textproto --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto --deployment-descriptor-path tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto --print-connectivity --hard-fail

  # these tests cover mux fixture as well
  TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3k*MeshGraphFabric2DDynamicTests*

  # GAP 1: Regression tests for pre-send teardown escape (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA=false).
  # FabricTeardownEscapeFixture verifies that FABRIC_2D init and teardown do not hang on T3K,
  # and that the can_send predicate + single pre-send teardown check prevent infinite TXQ spin.
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTeardownEscapeFixture.*"

  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_at_least_2x2_mesh.yaml
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml

  # Code profiling test
  TT_FABRIC_PROFILE_RX_CH_FWD=1 TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_code_profiling.yaml

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttnn_tests() {
  # Reset hardware state from any prior hung job.
  # Guard with timeout: tt-smi -r can itself hang indefinitely when hardware
  # (e.g. Device 4 ETH channels) is severely corrupted and needs a host reboot.
  # 120s is generous for a normal PCIe reset cycle; if it exceeds that we bail
  # rather than letting the whole test script hang forever.
  timeout 30 tt-smi -r || true
  # FIX GS-3 (#42429): warm-up open/close FABRIC_1D after initial tt-smi -r to clear
  # base-UMD state from non-MMIO channels.  Without this, the first GTest that opens
  # with FABRIC_2D sees fabric_stale_base_umd_channels_=true (FIX RZ), GTEST_SKIP()s,
  # record_test treats the skip as failure, issues another tt-smi -r, and the cycle
  # repeats until the hardware degrades into "failed to initialize FW!".
  # The warm-up mirrors FIX GS-2b in tests/nightly/t3000/ccl/conftest.py.
  # FIX TO (#42429): record wall-clock time before warm-up so we can detect the
  # ring-sync-timeout path (FIX TH2 extended timeout = 30s).  Normal warm-up
  # completes in <10s; a ring-sync timeout causes rescue_stuck_dispatch_cores to
  # hard-BRISC-reset dispatch ERISC cores (23-17, 19-17, 24-17, 20-16, 23-16) on
  # all 8 devices, leaving them in go_msg=0x02 stale state and corrupting the
  # subsequent topology check / FIX TL/TM recovery window.  A remedial tt-smi -r
  # immediately after the warm-up clears that stale state before the topology check.
  local WARM_START WARM_END WARM_DURATION WARM_OUTPUT WARM_RING_TIMEOUT
  WARM_START=$(date +%s)
  WARM_OUTPUT=$(python3 -u -c "
import sys, time
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] initial warm-up complete — base-UMD channels cleared for GTest')
except Exception as e:
    print(f'[FIX GS-3] WARNING: initial warm-up failed ({e}) — GTests may skip due to stale base-UMD', file=sys.stderr)
" 2>&1 || true)
  echo "$WARM_OUTPUT"
  WARM_END=$(date +%s)
  WARM_DURATION=$((WARM_END - WARM_START))
  # FIX UP (#42429): detect ring-sync timeout in warm-up output.
  # FIX TH3 extends timeout to 120s, so duration threshold moves to 120.
  # Additionally grep for Metal log markers that fire even when Python exits 0:
  #   "FIX TK"                    — teardown warning set when ring-sync timed out
  #   "ring_sync_already_timed_out" — guard variable logged in quiesce path
  #   "Timeout after.*ms.*master chan" — the actual timeout log from ring-sync poller
  # FIX DT-1: also detect dispatch-ERISC teardown timeout markers:
  #   "Timeout (N ms) waiting for physical cores" — dispatch ERISC teardown exceeded 1s
  #   "rescue of stuck dispatch cores" — Metal hard-reset ERISCs, leaving go_msg=0x02 stale
  # When detected, flag that the hardware is NOT ready for traffic.
  WARM_RING_TIMEOUT=0
  # FIX LM (#42429): cold-start relay-dead detection.
  # When FIX BX or FIX NZ fires for non-MMIO devices during warm-up, the ERISC relay
  # path (host MMIO chan=8 → remote chips 4-7) is dead BEFORE reset_cores begins.
  # This is hardware degradation (progressive link failure) that tt-smi -r cannot fix.
  # Detect this pattern so the FIX TL/TM recovery cycle is skipped — tt-smi -r
  # resets the chips but the relay stays dead, causing an identical 3-minute futile
  # retry before the same exit-1 is reached (observed: run 26052603107, t3k-05).
  WARM_RELAY_DEAD=0
  if echo "$WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX BX.*initialize_and_launch_firmware threw.*non-MMIO|FIX NZ.*skipping initialize_and_launch_firmware.*relay broken)"; then
    WARM_RELAY_DEAD=1
    echo "LOG_METAL: [FIX LM] cold-start relay-dead detected in warm-up (FIX BX/NZ fired for non-MMIO devices) — tt-smi -r cannot recover hardware degradation. (#42429)" >&2
  fi
  # [FIX UP-2]: Strip ANSI escape codes and CR before grepping — ANSI codes embedded in
  # WARM_OUTPUT (e.g. from Metal's coloured log output) corrupt grep pattern matching and
  # cause false negatives even when "FIX TK" is present in the raw output (run 26022229604).
  if echo "$WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
    echo "LOG_METAL: [FIX UP] ring-sync timeout marker detected in warm-up output — hardware not ready for traffic despite open/close exit 0." >&2
    WARM_RING_TIMEOUT=1
  fi
  if [[ $WARM_DURATION -ge 120 || $WARM_RING_TIMEOUT -eq 1 ]]; then
    echo "LOG_METAL: [FIX TO] warm-up ran ${WARM_DURATION}s (ring-sync timeout path, WARM_RING_TIMEOUT=${WARM_RING_TIMEOUT}). Running remedial tt-smi -r to clear dispatch-ERISC stale state. (#42429)"
    timeout 30 tt-smi -r || true
  fi

  # [FIX LM-2] (#42429): Fast-fail BEFORE GetNumAvailableDevices() when cold-start relay
  # dead was detected.  The Python topology-discovery call opens all MMIO devices and then
  # waits ETH_STARTUP_TIMEOUT (10 s) for each dead ETH core's heartbeat before throwing.
  # That 10s hang produces empty output → FIX TL-2 fires → tt-smi fallback also returns 0
  # → the log shows "0/8 chips" and the FIX LM check fires only afterwards.  Skipping the
  # Python call entirely when the relay is already known to be dead saves 10+ seconds and
  # produces a clean, attributable exit-1 without the misleading "0/8 chips" noise.
  if [[ $WARM_RELAY_DEAD -eq 1 ]]; then
    echo "LOG_METAL: [FIX LM] relay dead from cold start (FIX BX/NZ in warm-up) — skipping topology check + FIX TL/TM recovery (tt-smi -r cannot fix hardware link degradation)." >&2
    echo "LOG_METAL: ERROR — T3K topology degraded: ERISC relay path dead before reset_cores. Hardware needs engineer attention or host reboot." >&2
    exit 1
  fi

  # T3K topology sanity check — fail immediately if fewer than 8 chips are visible.
  # A degraded N300 host (FIX AQ path) shrinks the topology to 4 MMIO-only chips.
  # T3K multi-chip GTest fixtures call GTEST_SKIP() on a 4-chip topology and exit 0 —
  # CI then appears green even though every T3K-specific test was skipped.
  # This pre-check catches the degraded state before any test runs so CI reports a
  # real failure and the on-call engineer knows hardware needs attention.
  local n_chips raw_output
  # Use 2>&1 to capture UMD C++ stderr log messages — needed so that if Python crashes
  # its error output appears in raw_output and we can detect the crash rather than
  # silently treating empty output as 0 chips.  The numeric grep filters out non-numeric
  # lines (UMD log lines) from stdout+stderr.
  # Python crashes produce non-zero exit → with set -eo pipefail, the assignment itself
  # would abort the shell before reaching the n_chips guard below.  The || true
  # prevents that: a crash leaves raw_output with only error text → n_chips="" → handled below.
  # -u: unbuffered stdout so print(8) flushes immediately even if process aborts during teardown.
  # tr -d '\r': UMD loguru on some runners emits CRLF; grep '^[0-9]+$' fails on '8\r'.
  raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>&1) || true
  n_chips=$(echo "$raw_output" | tr -d '\r' | grep -E '^[0-9]+$' | tail -1) || true
  if [[ -z "$n_chips" ]]; then
    # [FIX TL-2]: Python crashed or produced no numeric output — try tt-smi as fallback
    # to count visible Wormhole chips.  This prevents a silent Python crash (e.g. stale
    # go_msg=0x02 from dispatch ERISC rescue) from being misread as 0 chips and
    # incorrectly triggering FIX TL topology recovery (run 26022229604).
    n_chips=$(tt-smi -ls 2>/dev/null | grep -c "Wormhole" || echo "0") || n_chips="0"
    echo "LOG_METAL: [FIX TL-2] GetNumAvailableDevices() returned empty (Python crashed?); tt-smi sees ${n_chips}/8 chips" >&2
  fi
  if ! [[ "$n_chips" =~ ^[0-9]+$ ]]; then
    echo "LOG_METAL: [FIX TL-0] T3K topology check returned non-numeric (${n_chips}); treating as 0 chips for recovery attempt" >&2
    n_chips="0"
  fi
  if [[ "$n_chips" -lt 8 ]]; then
    # FIX LM (#42429): if cold-start relay-dead was detected in the warm-up, skip the
    # FIX TL/TM recovery cycle entirely — tt-smi -r cannot restore a dead ERISC relay
    # caused by hardware degradation.  Failing fast saves 3+ minutes of futile retries
    # (2× warm-up cycles + 2× tt-smi -r) that all end with the same exit-1 outcome.
    if [[ $WARM_RELAY_DEAD -eq 1 ]]; then
      echo "LOG_METAL: [FIX LM] relay dead from cold start (FIX BX/NZ in warm-up) — skipping FIX TL/TM recovery (tt-smi -r cannot fix hardware link degradation)." >&2
      echo "LOG_METAL: ERROR — T3K topology degraded (${n_chips}/8 chips): ERISC relay path dead before reset_cores. Hardware needs engineer attention or host reboot." >&2
      exit 1
    fi
    # FIX TL (#42429): warm-up subprocess atexit can leave non-MMIO chips unreachable
    # (ARC timeout / relay-dead channels corrupt device visibility).
    # Attempt one recovery reset before failing the job entirely.
    echo "LOG_METAL: [FIX TL] T3K topology damaged after warm-up (${n_chips}/8 chips) — attempting recovery via tt-smi -r"
    timeout 30 tt-smi -r || true
    # FIX TM (#42429): after tt-smi -r, base-UMD firmware on non-MMIO ETH cores needs time to
    # initialise before topology discovery can reach them via relay.  A bare GetNumAvailableDevices()
    # call immediately after the reset races against firmware boot and returns only MMIO chips (4).
    # Re-run the full FIX GS-3 warm-up first: it opens/closes the mesh (which triggers relay
    # establish + base-UMD quiesce) so that the subsequent topology check sees all 8 chips.
    # FIX TM2 (#42429): capture FIX TM warm-up output and check for ring-sync timeout
    # markers, same as FIX UP does for the initial warm-up.  Without this, ring-sync
    # timeout inside FIX TM goes undetected: topology check sees 8 chips (relay works)
    # but ring never converged — first GTest hits stale_base_umd → SKIP loop.
    local TM_WARM_OUTPUT TM_RING_TIMEOUT
    TM_WARM_OUTPUT=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX TM] post-TL warm-up complete — relay re-established after recovery reset')
except Exception as e:
    print(f'[FIX TM] WARNING: post-TL warm-up failed ({e}) — topology check may still see degraded state', file=sys.stderr)
" 2>&1 || true)
    echo "$TM_WARM_OUTPUT"
    TM_RING_TIMEOUT=0
    if echo "$TM_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
      echo "LOG_METAL: [FIX TM2] ring-sync timeout detected in post-TL warm-up — hardware may not be ready for traffic." >&2
      TM_RING_TIMEOUT=1
    fi
    # FIX LM (#42429): also detect cold-start relay-dead in the FIX TM post-recovery warm-up.
    # If FIX BX fires again after tt-smi -r, the relay is permanently dead — fail fast.
    if echo "$TM_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX BX.*initialize_and_launch_firmware threw.*non-MMIO|FIX NZ.*skipping initialize_and_launch_firmware.*relay broken)"; then
      echo "LOG_METAL: [FIX LM] cold-start relay-dead PERSISTS after tt-smi -r (FIX BX/NZ in post-TL warm-up) — hardware cannot be recovered by reset. (#42429)" >&2
      echo "LOG_METAL: ERROR — T3K topology still degraded: ERISC relay dead after recovery. Hardware needs engineer attention or host reboot." >&2
      exit 1
    fi
    raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>&1) || true
    n_chips=$(echo "$raw_output" | tr -d '\r' | grep -E '^[0-9]+$' | tail -1) || true
    if [[ -z "$n_chips" ]]; then n_chips="0"; fi
    if [[ "$n_chips" -lt 8 ]]; then
      echo "LOG_METAL: ERROR — T3K topology still degraded after recovery: ${n_chips}/8 chips visible." >&2
      echo "LOG_METAL: Hardware needs host reboot or engineer attention." >&2
      exit 1
    fi
    echo "LOG_METAL: [FIX TL/TM] topology recovered: ${n_chips}/8 chips visible after reset+warm-up."
  fi
  echo "LOG_METAL: T3K topology OK — ${n_chips}/8 chips visible."

  # FIX UP2 (#42429): Pre-test-loop ring-sync health gate.
  # If EITHER the initial warm-up (FIX UP) OR the FIX TM recovery warm-up detected
  # ring-sync timeout, the hardware is not ready for traffic even though topology shows
  # 8 chips.  Run one more remedial tt-smi -r + warm-up and check again.  If ring sync
  # STILL fails after this second attempt, abort with INFRA_ERROR — the cluster is too
  # degraded for meaningful test execution and needs a reboot.
  if [[ ${WARM_RING_TIMEOUT:-0} -eq 1 || ${TM_RING_TIMEOUT:-0} -eq 1 ]]; then
    echo "LOG_METAL: [FIX UP2] ring-sync timeout detected in pre-test warm-up — attempting one more reset+warm-up before test loop." >&2
    timeout 30 tt-smi -r || true
    local UP2_WARM_OUTPUT UP2_RING_TIMEOUT
    UP2_WARM_OUTPUT=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX UP2] pre-test-loop warm-up complete')
except Exception as e:
    print(f'[FIX UP2] WARNING: warm-up failed ({e})', file=sys.stderr)
" 2>&1 || true)
    echo "$UP2_WARM_OUTPUT"
    UP2_RING_TIMEOUT=0
    if echo "$UP2_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
      UP2_RING_TIMEOUT=1
    fi
    if [[ $UP2_RING_TIMEOUT -eq 1 ]]; then
      # FIX UP3 (#42429): The warm-up itself triggers the dispatch-ERISC timeout cycle:
      # base-UMD ERISC channels (loaded after tt-smi -r) interfere with dispatch FW
      # during open/close, causing rescue_stuck_dispatch_cores to fire and leave
      # go_msg=0x02 stale.  Running more warm-ups cannot break this circular dependency.
      # Instead: do a final tt-smi -r to clear the go_msg=0x02 stale state left by
      # rescue_stuck_dispatch_cores, then proceed WITHOUT another warm-up.
      # The test sessions have FIX SC (stale go_msg cleanup) + FIX M/RZ2 (base-UMD
      # transition + ring-sync health check) to handle first-session state safely.
      echo "LOG_METAL: [FIX UP3] dispatch-ERISC timeout loop detected — running final tt-smi -r to clear rescue_stuck stale state, then proceeding to tests without warm-up. (#42429)" >&2
      timeout 30 tt-smi -r || true
    else
      echo "LOG_METAL: [FIX UP2] ring-sync healthy after retry — proceeding to test loop."
    fi
  fi

  # Per-test-failure hardware reset hook.
  # Call immediately after each test line: `cmd; record_test`
  # Captures $? from the preceding command, accumulates into $fail, and
  # triggers tt-smi -r on any individual failure so subsequent tests start
  # from a clean hardware state rather than inheriting stale ERISC/ETH residue.
  #
  # Skip-count guard: if a GTest binary wrote /tmp/gtest_last_result.xml (via
  # GTEST_OUTPUT env var), parse the skip count and treat non-zero skips as failure.
  # This catches the case where T3K fixtures call GTEST_SKIP() when topology is
  # degraded mid-run but exit 0 (GTest exits 0 on all-skipped by default).
  record_test() {
    local rc=$?
    # Check GTest XML for skip-only passes if the XML output file is present.
    # GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" must be set before each GTest invocation.
    if [[ $rc -eq 0 && -f /tmp/gtest_last_result.xml ]]; then
      local total_skipped
      total_skipped=$(grep -oP '(?<=skipped=")[0-9]+' /tmp/gtest_last_result.xml \
                        | awk '{s+=$1} END {print s+0}' 2>/dev/null || echo 0)
      if [[ "${total_skipped:-0}" -gt 0 ]]; then
        echo "LOG_METAL: ERROR — exit 0 but ${total_skipped} test(s) SKIPPED in GTest XML." >&2
        echo "LOG_METAL: T3K topology may have degraded mid-run — treating as failure." >&2
        rc=1
      fi
      rm -f /tmp/gtest_last_result.xml
    fi
    fail+=$rc
    if [[ $rc -ne 0 ]]; then
      echo "LOG_METAL: test returned rc=$rc — resetting hardware via tt-smi"
      timeout 30 tt-smi -r || true
      # FIX GS-3 (#42429): warm-up after per-test reset to prevent base-UMD reset cycle.
      # After tt-smi -r, non-MMIO ETH channels reload base-UMD firmware. If the next
      # GTest opens with FABRIC_2D without this warm-up, FIX M transitions the channels
      # but sets fabric_stale_base_umd_channels_=true, causing GTEST_SKIP() → another
      # tt-smi -r → loop until hardware cannot initialize FW at all.
      local post_warm_output post_ring_timeout
      post_warm_output=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] post-reset warm-up complete — base-UMD channels cleared')
except Exception as e:
    print(f'[FIX GS-3] WARNING: post-reset warm-up failed ({e}) — next test may still skip', file=sys.stderr)
" 2>&1 || true)
      echo "$post_warm_output"
      # FIX UP (#42429): detect ring-sync timeout in post-reset warm-up output.
      # If Metal logs "FIX TK" or the ring-sync poller timeout message, the ring never
      # completed — hardware is NOT ready for traffic even though Python exited 0.
      # Increment consecutive_ring_timeout; after 3 in a row abort with INFRA_ERROR
      # so CI marks the job failed rather than looping indefinitely.
      post_ring_timeout=0
      if echo "$post_warm_output" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
        post_ring_timeout=1
      fi
      if [[ $post_ring_timeout -eq 1 ]]; then
        consecutive_ring_timeout=$((consecutive_ring_timeout + 1))
        echo "LOG_METAL: [FIX UP] post-reset warm-up ring-sync timeout #${consecutive_ring_timeout}/3 — hardware not ready for traffic." >&2
        if [[ $consecutive_ring_timeout -ge 3 ]]; then
          echo "LOG_METAL: [FIX UP] INFRA_ERROR — ring-sync timeout on ${consecutive_ring_timeout} consecutive warm-ups. Hardware requires reboot. Aborting test run." >&2
          exit 1
        fi
      else
        consecutive_ring_timeout=0
      fi
    fi
  }

  # Record the start time
  fail=0
  consecutive_ring_timeout=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttnn_tests"

  # ===========================================================================
  # BATCH 1: AllGather — runs FIRST to keep the primary goal in focus.
  # The mission of this branch is to eliminate AllGather hangs. Any regression
  # that reintroduces a hang should surface before any other test batch runs.
  # ===========================================================================
  echo "LOG_METAL: [BATCH 1/3] AllGather tests"

  # chip-3 CQ0 AllGather hang reproducer — runs the async_cq0 binary solo
  # (predecessor unit_tests_ttnn_ccl_ops runs below as part of the CCL batch).
  # With TT_METAL_DISABLE_ASYNC_CQ0_T3K_TEMP set, AsyncExecutionWorksCQ0 will
  # GTEST_SKIP() rather than hang. If the env var is ever removed this is the
  # canary that catches the regression immediately.
  # See tests/scripts/t3000/repro_ccl_cq0_hang.sh for full repro instructions.
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" ${TT_METAL_HOME}/tests/scripts/t3000/repro_ccl_cq0_hang.sh --solo ; record_test

  # CCL operation binaries — AllGather, ReduceScatter, and multi-tensor CCL.
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl ; record_test
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl_multi_tensor ; record_test
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl_ops ; record_test

  # test_ccl_multi_cq_multi_device: chip-3 CQ0 AllGather hang investigation.
  # - Split each TEST_F into its own subprocess so predecessor state cannot bleed
  #   across and so a hang pinpoints exactly one test.
  # - Brief sleep between the preceding FABRIC_2D binary and this FABRIC_1D binary
  #   gives chips that were slow to drain TERMINATED a chance to settle before the
  #   next fabric bring-up. If removing this sleep makes the hang reappear, that
  #   points at residual device state (H-A in the investigation plan).
  # - Outer `timeout` intentionally omitted here so the in-process
  #   TT_METAL_OPERATION_TIMEOUT_SECONDS + hang_report.py triage hook can fire and
  #   capture dispatcher/worker state before the process is killed. A much looser
  #   ceiling is provided via `timeout 600` as a last-resort backstop.
  sleep 2
  for ccl_mcq_test in \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0" \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0CQ1" \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksMultithreadCQ0"; do
      echo "LOG_METAL: running test_ccl_multi_cq_multi_device --gtest_filter=${ccl_mcq_test}"
      GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 600 ./build/test/ttnn/test_ccl_multi_cq_multi_device --gtest_filter="${ccl_mcq_test}"
      record_test
      sleep 1
  done

  # AllGather-specific GAP regression tests.
  # GAP-21: Rapid AllGather+quiesce stress (FIX AE/AF/AN)
  # Explicit reset and settle before GAP-21 stress test — prior tests may leave hardware in marginal state
  tt-smi -r || true
  sleep 10
  # FIX GS-3: warm-up after explicit reset to clear base-UMD state before Python conftest runs.
  python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] pre-GAP21 warm-up complete — base-UMD channels cleared')
except Exception as e:
    print(f'[FIX GS-3] WARNING: pre-GAP21 warm-up failed ({e})', file=sys.stderr)
" 2>&1 || true
  sleep 5
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap21_rapid_allgather_quiesce_stress.py::test_rapid_allgather_quiesce_stress ; record_test
  # GAP-22: AllGather interrupted mid-flight by mesh close (FIX AO/AP/AD)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap22_allgather_inflight_close.py::test_allgather_inflight_close ; record_test
  # GAP-23: Partial-mesh quiesce cycling with AllGather (FIX AK/AM/AE)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap23_partial_mesh_quiesce_cycling.py::test_partial_mesh_quiesce_cycling ; record_test
  # GAP-25: Back-to-back AllGather without explicit sync (FIX AE/AF)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap25_back_to_back_allgather_nosync.py::test_back_to_back_allgather_nosync ; record_test
  # GAP-38: AllGather correctness after FIX BA teardown chain (FIX BA + FIX AC + FIX AY).
  # GAP-37 verifies the second open is FAST. GAP-38 verifies the second session AllGather
  # produces numerically correct output (PCC >= 0.9999). These are orthogonal: a regression
  # where FIX BA cleans up timing but leaves stale EDM routing tables would pass GAP-37 but
  # fail GAP-38 (wrong AllGather output or hang).
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap38_fixba_allgather_correctness_after_cleanup.py::test_gap38_fixba_allgather_correctness_after_cleanup ; record_test

  # ===========================================================================
  # BATCH 2: Broader TTNN and TT-Metal tests.
  # ===========================================================================
  echo "LOG_METAL: [BATCH 2/3] Broader TTNN tests"

  # GTEST_OUTPUT writes skip counts to XML so record_test can detect all-skipped passes.
  # FIX RC: Skip MeshDevice1x4FabricFixture.TestGenericOpAllGather on T3K — it hits the
  # same dispatch hang as AsyncExecutionWorksCQ0 (unsafe NOC access at 0x880030060 on
  # non-MMIO chips).  The skip guard already exists in test_generic_op.cpp:1221; we just
  # need to set the env var so it fires.
  TT_METAL_DISABLE_ASYNC_CQ0_T3K_TEMP=1 GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 900 ./build/test/ttnn/unit_tests_ttnn ; record_test
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 600 ./build/test/ttnn/unit_tests_ttnn_tensor ; record_test
  # Disabled: ManualPagesIterationInterleaved rank_6+ hangs with unsafe NOC read on T3K (issue #42195)
  # timeout 300 ./build/test/ttnn/unit_tests_ttnn_accessor ; record_test
  pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace.py ; record_test
  pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_events.py ; record_test
  pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher.py::test_run_prefetcher_post_commit_multi_device ; record_test
  pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py ; record_test
  pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_async.py ; record_test
  pytest tests/ttnn/distributed/test_tensor_parallel_example_T3000.py ; record_test
  pytest tests/ttnn/distributed/test_data_parallel_example.py ; record_test
  pytest tests/ttnn/distributed/test_hybrid_data_tensor_parallel_example_T3000.py ; record_test

  # ===========================================================================
  # BATCH 3: ETH/ERISC infrastructure regression tests.
  # Targeted async-dispatch + teardown race condition regression tests.
  # Validates fixes for the ERISC stale firmware race (AI-JOURNAL.md Pass A-F).
  # ===========================================================================
  echo "LOG_METAL: [BATCH 3/3] ETH/ERISC infrastructure regression tests"

  # Scenario D (Fabric2DAsyncDispatchThenReinit) exercises the ETH-router
  # TERMINATED poll in FabricFirmwareInitializer::teardown() — the code path
  # that Scenarios A/B/C (FabricConfig::DISABLED) bypass entirely.
  # Scenario E (RepeatedFabric2DTeardownCycles) stress-tests 2 consecutive
  # FABRIC_2D open/close cycles to catch accumulated ERISC state (CI Iters 3-5).
  # Scenarios J/K (AsyncTeardownFabric1DQuiesceFixture) test FABRIC_1D quiesce_devices()
  # with has_tensix_mux=false — the iter12 regression path that Scenarios H/I miss.
  # Scenario L (AsyncTeardownFabric2DRepeatFixture.Fabric2DSlowKernelTeardownRace) fills
  # the gap between Scenario F (slow kernel, no ERISC) and Scenario D (ERISC, blank kernel):
  # FABRIC_2D + busy_spin = both ERISC EDM and BRISC active when close() fires.
  # Scenario M (AsyncTeardownKillPredecessorFixture) is the CRITICAL missing test:
  # fork()+SIGKILL simulates predecessor test being killed; ERISCs left in ACTIVE state;
  # parent re-opens → terminate_stale_erisc_routers() ACTIVE path exercised for the first
  # time. This is the exact CI failure scenario the fix was written to handle. (+15s wait)
  # FabricFirmwareInitializer: compile-time enum coverage check (no device required).
  # QuiesceStressFixture: 5-cycle FABRIC_2D quiesce stress (Scenario AB).
  # PhaseWFixture: FIX W regression — all-dead MMIO clean-return invariant.
  # PhaseZFixture: FIX Z regression — relay-broken CQ fast-throw accessor check.
  # FixAvRelayBrokenSysmemGuardFixture: GAP-28 — FIX AV relay-broken guard in
  #   configure_command_queue_programs prevents hang on dead relay sysmem reset.
  # ClusterTeardownHangRelayBrokenFixture: GAP-29 — FIX AW ~Cluster doesn't hang
  #   in wait_for_non_mmio_flush after relay-broken quiesce (FIX AC PCIe reset).
  # FixAyDeferredNonMmioResetFixture: GAP-31 — FIX AY deferred non-MMIO ERISC
  #   reset after FIX AC restores MMIO relay; second MeshDevice::create() must
  #   not hang on write_non_mmio with FABRIC-firmware non-MMIO ERISCs.
  # FixAzL1BarrierSkipNoPriorFabricFixture: GAP-32 — FIX AZ l1_barrier not called
  #   after assert_cores throws for non-MMIO when relay_broken_non_mmio is empty
  #   (no FabricFirmwareInitializer session — e.g. unit_tests_ttnn_udm scenario).
  # EthCoordPreservedOnAqSkipFixture: GAP-41 — FIX NT EthCoord preserved in
  #   chip_locations for FIX-AQ-skipped chips (no TT_FATAL on YAML coord lookup).
  # MmioEthCoordBeforeRelayGuardFixture: GAP-42 — FIX NU MMIO EthCoord captured
  #   via PCIe before FIX W heartbeat guard loop (no TT_FATAL when all ETH dead).
  # AsyncBuildPhaseRelayGuardFixture: GAP-43 — FIX NV + FIX NW: non-MMIO chips
  #   skipped for get_device_aiclk and clear_launch_messages_on_eth_cores in
  #   run_async_build_phase; dead relay must NOT throw through async futures.
  # WriteCorRelayGuardFixture: GAP-44 — FIX NX: write_core() for non-MMIO chips
  #   wraps both write_to_device + wait_for_non_mmio_flush in one try/catch so
  #   relay timeout in set_internal_routing_info or WatcherServer::init_devices
  #   does not propagate to MetalContext::initialize as an uncaught exception.
  # EthTrainingFabricEriscsFixture: GAP-45 — FIX X extension: wait_eth_core_training
  #   now also returns early when heartbeat IS 0xABCDxxxx but training stays
  #   IN_PROGRESS after 2000ms — fabric firmware left ETH_TRAIN_STATUS_ADDR=0 via
  #   ConfigureDeviceWithProgram .bss write; original FIX X only skipped no-heartbeat
  #   case, missing the "fabric ERISC alive but training never written" case.
  # RelayBrokenChipsCacheFixture: GAP-46 — FIX NY: relay_broken_chips_ cache in
  #   Cluster::write_core() eliminates per-channel 5s UMD timeout stall after the
  #   first failure for a dead-relay chip. Without FIX NY, FIX NX alone allows
  #   set_internal_routing_info_for_ethernet_cores to stall 6×5s=30s per chip
  #   (6 ETH channels × 5s UMD timeout each). FIX NY caches the first failure in
  #   relay_broken_chips_ so channels 2-6 return immediately (0ms). Primary check
  #   is TIMING (35s budget); FIX NX regression shows as exit non-zero.
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 900 ./build/test/tt_metal/distributed/distributed_unit_tests \
    --gtest_filter='Gap1ThreePassEthLaunchFixture.*:LaunchGateLiveEriscFixture.*:ERISCHeartbeatFixture.*:PartialMeshQuiesceFixture.*:RelayBrokenTeardownFixture.*:Phase25RelayBrokenCascadeFixture.*:MmioPhase5RelayBrokenFixture.*:InitRouterSyncDeadRelayFixture.*:ParallelHeartbeatPollFixture.*:ChannelsNotReadyLifecycleFixture.*:RelayTimeoutToleranceFixture.*:TeardownReopenEthOrderingFixture.*:AsyncTeardownRaceFixture.*:AsyncTeardownMultiCQFixture.*:AsyncTeardownFabric2DFixture.*:AsyncTeardownFabric2DRepeatFixture.*:AsyncTeardownFabric1DQuiesceFixture.*:AsyncTeardownKillPredecessorFixture.*:FabricFirmwareInitializer.*:QuiesceStressFixture.*:PhaseWFixture.*:PhaseZFixture.*:FixAvRelayBrokenSysmemGuardFixture.*:ClusterTeardownHangRelayBrokenFixture.*:FixAyDeferredNonMmioResetFixture.*:FixAzL1BarrierSkipNoPriorFabricFixture.*:EthCoordPreservedOnAqSkipFixture.*:MmioEthCoordBeforeRelayGuardFixture.*:AsyncBuildPhaseRelayGuardFixture.*:WriteCorRelayGuardFixture.*:EthTrainingFabricEriscsFixture.*:RelayBrokenChipsCacheFixture.*:ReadCoreRelayGuardFixture.*:FwLaunchAddrForceResetFixture.*:FwLaunchAddrRescueFixture.*:FwLaunchAddrQuiesceFixture.*:TeardownNullControlPlaneFixture.*:UmdHeartbeatSkipExitFixture.*:Phase25RelayRetryFixture.*:FixE2AyProbeDeadFayTriggerFixture.*:FixM2DeadPeerEriscResetFixture.*:FixPlBarrierGuardDeadRelayFixture.*:FixQcNonMmioResetCoresSkipFixture.*:FixQbResetLoopEarlyBreakFixture.*:FixPyPzPhase25TopologyTimeoutFixture.*:FixQdDeadRouterMmioSkipFixture.*:FixQuReassertFlagsFixture.*:FixNyRelayMuxClusterGuardFixture.*:FixTbTopologyMapperUnknownAsicFixture.*:FixQvPhase4SkipFixture.*:FixRzStaleBaseUmdFlagFixture.*:FixTf2dFabricHeaderArgsGuardFixture.*:FixTgControlPlaneHostRankGuardFixture.*:FixThRelayMuxNoLinksGuardFixture.*:FixTkDegradedClusterChipFilterFixture.*:FixTlDegradedClusterBailBeforeCreateMeshesFixture.*:FixTiRingSyncFastSkipFixture.*:Gap74BaseFixtureTeardownGuardFixture.*:FixQwbStaleBaseUmdSkipGuardFixture.*:FixTg2BaseUmdPartialL1ClearFixture.*:FixPhYamlEthCoordGracefulSkip.*:Gap75SetUpNotActiveSkipGuard.*:Gap76FixSbIdleEthInitFabricGuard.*' ; record_test

  # Remaining GAP regression tests — infrastructure / ETH hang fixes.
  # GAP-24: Rapid mesh close/reopen cycling under FABRIC_2D (FIX AD/AC/AL/AQ)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap24_rapid_close_reopen_cycling.py::test_rapid_close_reopen_cycling ; record_test
  # GAP-26: FIX AS canary poll timeout → newly-dead graceful degradation (FIX AS sad-path)
  timeout 180 pytest -svv tests/nightly/t3000/ccl/test_gap26_fixas_canary_timeout_graceful.py::test_gap26_fixas_canary_timeout_graceful ; record_test
  # GAP-27: FIX AV — non-MMIO sysmem_manager reset prevents stale in-flight counter
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap27_fixav_nonmmio_sysmem_reset.py::test_gap27_fixav_nonmmio_sysmem_reset ; record_test
  # GAP-30: FIX AL — STARTED early-exit timing (kStartedTimeoutMs=3000ms) bounds quiesce wait
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap30_fixal_started_early_exit_timing.py::test_gap30_fixal_started_early_exit_timing ; record_test
  # GAP-34: FIX AM — Phase 5b skipped when master chan at STARTED (out-of-mesh peer);
  #   saves ~2s per device vs FIX AL alone; caught TestMeshWidthShardedCopy3D timeout in CI run 25048641877
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap34_fixam_phase5b_skip_timing.py::test_gap34_fixam_phase5b_skip_timing ; record_test
  # GAP-35: FIX AT — Phase 5 handshake poll skipped when MMIO master chan was FIX AS Pass-0
  #   timeout'd (WH BRISC boot >500ms → status=0x0, no firmware loaded). Without FIX AT:
  #   Phase 5 polls for 10s per MMIO device = 20s overhead per cycle (2 MMIO devices on T3K).
  #   Caught AsyncExecutionWorksCQ0 timeout in CI run 25054499947.
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap35_fixat_fixas_mmio_phase5_skip.py ; record_test
  # GAP-36: FIX AV — device_relay_dead per-device early-exit in FIX AY loop when relay is
  #   dead after FIX AC PCIe-reset. Without FIX AV: 4 ETH cores × 2 non-MMIO devices × 5s
  #   timeout = 40s teardown → CI SIGALRM. With FIX AV: 1 throw × 2 devices × 5s = 10s.
  #   Root CI failure: run 25060970918 (job 73417098227).
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap36_fixav_relay_dead_per_device_skip.py::test_gap36_fixav_relay_dead_per_device_skip ; record_test
  # GAP-37: FIX BA — STARTED-state non-MMIO devices must be cleaned up at teardown.
  #   Without FIX BA: FIX AM sets channels_not_ready=true but relay_broken=false, so
  #   teardown Step 1 skips these devices. Non-MMIO ERISCs remain in FABRIC STARTED state.
  #   Next session's topology discovery (create_remote_device → read_non_mmio) stalls 5s
  #   per device → ALL subsequent tests fail (observed: run 25066686656, all 359 failed).
  #   With FIX BA: STARTED-state devices added to relay_broken_non_mmio → FIX AC + FIX AY
  #   clean up ERISCs. Second open in this test must complete < 15s.
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap37_fixba_started_state_nonmmio_cleanup.py::test_gap37_fixba_started_state_nonmmio_cleanup ; record_test
  # GAP-39: FIX NS — Single topology discovery per open.
  # Verifies that MetalEnvImpl::initialize_base_objects() does NOT trigger a redundant
  # topology discovery before Cluster creation, which would fill relay queues to 4/4
  # capacity on systems with stale FABRIC-mode ERISCs (14m40s hang observed in CI).
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap39_fixns_single_topology_discovery.py::test_gap39_fixns_single_topology_discovery ; record_test
  # GAP-40: FIX AE — Catch flush timeouts in write_core/write_reg/noc_multicast_write
  # and pre-mark remote chips relay-broken in ~Cluster() before close_device().
  # Verifies: (1) no 5s-per-call cascade from dead-relay mid-session writes; and
  # (2) no heap corruption from racing ~Cluster() + Cluster() UMD global state access
  # (FIX AE supersedes FIX AW background-thread approach).
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap40_fixae_flush_timeout_catch.py::test_gap40_fixae_flush_timeout_catch ; record_test

  # Additional GAP regression tests — previously added to tests/ but missing from runner.
  # GAP-11: Phase 2.5 force-reset + Pass-0 canary chain validation (FIX AS/PG)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_phase25_force_reset_pass0_chain.py::test_phase25_force_reset_pass0_chain ; record_test
  # GAP-12: Cross-device relay_broken race in Phase 5 concurrent quiesce (FIX AN)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_cross_device_relay_broken_race.py::test_cross_device_relay_broken_race ; record_test
  # GAP-13: L1 corruption cascade across 3+ sequential sessions (FIX AE/AC)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_l1_corruption_cascade.py::test_l1_corruption_cascade_broken ; record_test
  # GAP-14: Phase 4 MUX timeout force-reset does not leave device in bad state (FIX AC)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_phase4_mux_timeout_recovery.py::test_phase4_mux_timeout_recovery ; record_test
  # GAP-15: ENTRY snapshot deadline false positive does not permanently degrade fabric
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_snapshot_deadline_false_positive.py::test_snapshot_deadline_false_positive ; record_test
  # GAP-16: 3-pass ETH launch ordering prevents simultaneous handshake deadlock (FIX AE/AF)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_3pass_eth_launch_ordering.py::test_3pass_eth_launch_no_deadlock ; record_test
  # GAP-17: Partial-mesh quiesce — FIX AK non-fatal REMOTE_HANDSHAKE_COMPLETE
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_partial_mesh_handshake_tolerance.py::test_partial_mesh_quiesce_nonfatal ; record_test
  # GAP-18: Router sync graceful degradation on dead-relay neighbor (FIX AL)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_router_sync_graceful_degradation.py::test_router_sync_no_sigabrt ; record_test
  # GAP-19: channels_not_ready_for_traffic_ flag cleared on re-init (FIX AM lifecycle)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_channels_not_ready_flag_lifecycle.py::test_channels_not_ready_cleared_on_reinit ; record_test
  # GAP-20: Teardown parallel heartbeat poll converges within budget (FIX AD)
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_teardown_parallel_heartbeat_poll.py::test_teardown_heartbeat_poll_parallel ; record_test
  # GAP-79: FIX XY-2 — relay_broken cleared after ERISC force-reset → AllGather correctness
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap79_fixxy2_relay_broken_cleared_allgather.py::test_relay_broken_cleared_allgather_correctness ; record_test
  # GAP-80: FIX DT-1 — dispatch ERISC teardown timeout in warm-up triggers remedial tt-smi -r
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap80_fixdt1_dispatch_erisc_timeout_warmup_allgather.py::test_gap80_fixdt1_dispatch_erisc_timeout_warmup_allgather ; record_test
  # GAP-81: FIX UP3 — dispatch-ERISC timeout loop in warm-up → skip warm-up path
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap81_fixup3_dispatch_erisc_loop_warmup_skip_allgather.py::test_gap81_fixup3_dispatch_erisc_loop_warmup_skip_allgather ; record_test
  # GAP-82: FIX SC-ADDR — ETH cores in not_done_cores use per-core-type go_msg address
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap82_fixscaddr_eth_not_done_cores_allgather.py::test_gap82_fixscaddr_eth_not_done_cores_allgather ; record_test
  # GAP-83: FIX RR/RS — MMIO ROM-postcode channel recovery → recovered channels load firmware → AllGather succeeds
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap83_fixrr_rs_mmio_rompostcode_recovered_allgather.py::test_gap83_fixrr_rs_mmio_rompostcode_recovered_allgather ; record_test

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttnn_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_racecondition_hunt_tests() {
  # Race condition investigation suite — BATCH 1 (AllGather canary) + BATCH 3
  # (ETH/ERISC regression). Omits BATCH 2 (general TTNN) which is unrelated to
  # the branch goal. Runs all tests added by nsexton/0-racecondition-hunt.
  # Reset hardware state from any prior hung job.
  # (mirrors run_t3000_ttnn_tests preamble)
  timeout 30 tt-smi -r || true
  local WARM_START WARM_END WARM_DURATION WARM_OUTPUT WARM_RING_TIMEOUT
  WARM_START=$(date +%s)
  WARM_OUTPUT=$(python3 -u -c "
import sys, time
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] initial warm-up complete — base-UMD channels cleared for GTest')
except Exception as e:
    print(f'[FIX GS-3] WARNING: initial warm-up failed ({e}) — GTests may skip due to stale base-UMD', file=sys.stderr)
" 2>&1 || true)
  echo "$WARM_OUTPUT"
  WARM_END=$(date +%s)
  WARM_DURATION=$((WARM_END - WARM_START))
  WARM_RING_TIMEOUT=0
  # FIX LM (#42429): cold-start relay-dead detection.
  # When FIX BX or FIX NZ fires for non-MMIO devices during warm-up, the ERISC relay
  # path (host MMIO chan=8 → remote chips 4-7) is dead BEFORE reset_cores begins.
  # This is hardware degradation (progressive link failure) that tt-smi -r cannot fix.
  # Detect this pattern so the FIX TL/TM recovery cycle is skipped — tt-smi -r
  # resets the chips but the relay stays dead, causing an identical 3-minute futile
  # retry before the same exit-1 is reached (observed: run 26052603107, t3k-05).
  WARM_RELAY_DEAD=0
  if echo "$WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX BX.*initialize_and_launch_firmware threw.*non-MMIO|FIX NZ.*skipping initialize_and_launch_firmware.*relay broken)"; then
    WARM_RELAY_DEAD=1
    echo "LOG_METAL: [FIX LM] cold-start relay-dead detected in warm-up (FIX BX/NZ fired for non-MMIO devices) — tt-smi -r cannot recover hardware degradation. (#42429)" >&2
  fi
  # [FIX UP-2]: Strip ANSI escape codes and CR before grepping — ANSI codes embedded in
  # WARM_OUTPUT (e.g. from Metal's coloured log output) corrupt grep pattern matching and
  # cause false negatives even when "FIX TK" is present in the raw output (run 26022229604).
  if echo "$WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
    echo "LOG_METAL: [FIX UP] ring-sync timeout marker detected in warm-up output — hardware not ready for traffic despite open/close exit 0." >&2
    WARM_RING_TIMEOUT=1
  fi
  if [[ $WARM_DURATION -ge 120 || $WARM_RING_TIMEOUT -eq 1 ]]; then
    echo "LOG_METAL: [FIX TO] warm-up ran ${WARM_DURATION}s (ring-sync timeout path, WARM_RING_TIMEOUT=${WARM_RING_TIMEOUT}). Running remedial tt-smi -r to clear dispatch-ERISC stale state. (#42429)"
    timeout 30 tt-smi -r || true
  fi

  # [FIX LM-2] (#42429): Fast-fail BEFORE GetNumAvailableDevices() when cold-start relay
  # dead was detected.  The Python topology-discovery call opens all MMIO devices and then
  # waits ETH_STARTUP_TIMEOUT (10 s) for each dead ETH core's heartbeat before throwing.
  # That 10s hang produces empty output → FIX TL-2 fires → tt-smi fallback also returns 0
  # → the log shows "0/8 chips" and the FIX LM check fires only afterwards.  Skipping the
  # Python call entirely when the relay is already known to be dead saves 10+ seconds and
  # produces a clean, attributable exit-1 without the misleading "0/8 chips" noise.
  if [[ $WARM_RELAY_DEAD -eq 1 ]]; then
    echo "LOG_METAL: [FIX LM] relay dead from cold start (FIX BX/NZ in warm-up) — skipping topology check + FIX TL/TM recovery (tt-smi -r cannot fix hardware link degradation)." >&2
    echo "LOG_METAL: ERROR — T3K topology degraded: ERISC relay path dead before reset_cores. Hardware needs engineer attention or host reboot." >&2
    exit 1
  fi

  local n_chips raw_output
  raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>&1) || true
  n_chips=$(echo "$raw_output" | tr -d '\r' | grep -E '^[0-9]+$' | tail -1) || true
  if [[ -z "$n_chips" ]]; then
    # [FIX TL-2]: Python crashed or produced no numeric output — try tt-smi as fallback
    # to count visible Wormhole chips.  This prevents a silent Python crash (e.g. stale
    # go_msg=0x02 from dispatch ERISC rescue) from being misread as 0 chips and
    # incorrectly triggering FIX TL topology recovery (run 26022229604).
    n_chips=$(tt-smi -ls 2>/dev/null | grep -c "Wormhole" || echo "0") || n_chips="0"
    echo "LOG_METAL: [FIX TL-2] GetNumAvailableDevices() returned empty (Python crashed?); tt-smi sees ${n_chips}/8 chips" >&2
  fi
  if ! [[ "$n_chips" =~ ^[0-9]+$ ]]; then
    echo "LOG_METAL: [FIX TL-0] T3K topology check returned non-numeric (${n_chips}); treating as 0 chips for recovery attempt" >&2
    n_chips="0"
  fi
  if [[ "$n_chips" -lt 8 ]]; then
    # FIX LM (#42429): if cold-start relay-dead was detected in the warm-up, skip the
    # FIX TL/TM recovery cycle entirely — tt-smi -r cannot restore a dead ERISC relay
    # caused by hardware degradation.  Failing fast saves 3+ minutes of futile retries
    # (2× warm-up cycles + 2× tt-smi -r) that all end with the same exit-1 outcome.
    if [[ $WARM_RELAY_DEAD -eq 1 ]]; then
      echo "LOG_METAL: [FIX LM] relay dead from cold start (FIX BX/NZ in warm-up) — skipping FIX TL/TM recovery (tt-smi -r cannot fix hardware link degradation)." >&2
      echo "LOG_METAL: ERROR — T3K topology degraded (${n_chips}/8 chips): ERISC relay path dead before reset_cores. Hardware needs engineer attention or host reboot." >&2
      exit 1
    fi
    echo "LOG_METAL: [FIX TL] T3K topology damaged after warm-up (${n_chips}/8 chips) — attempting recovery via tt-smi -r"
    timeout 30 tt-smi -r || true
    local TM_WARM_OUTPUT TM_RING_TIMEOUT
    TM_WARM_OUTPUT=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX TM] post-TL warm-up complete — relay re-established after recovery reset')
except Exception as e:
    print(f'[FIX TM] WARNING: post-TL warm-up failed ({e})', file=sys.stderr)
" 2>&1 || true)
    echo "$TM_WARM_OUTPUT"
    TM_RING_TIMEOUT=0
    if echo "$TM_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
      echo "LOG_METAL: [FIX TM2] ring-sync timeout detected in post-TL warm-up — hardware may not be ready for traffic." >&2
      TM_RING_TIMEOUT=1
    fi
    # FIX LM (#42429): also detect cold-start relay-dead in the FIX TM post-recovery warm-up.
    # If FIX BX fires again after tt-smi -r, the relay is permanently dead — fail fast.
    if echo "$TM_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX BX.*initialize_and_launch_firmware threw.*non-MMIO|FIX NZ.*skipping initialize_and_launch_firmware.*relay broken)"; then
      echo "LOG_METAL: [FIX LM] cold-start relay-dead PERSISTS after tt-smi -r (FIX BX/NZ in post-TL warm-up) — hardware cannot be recovered by reset. (#42429)" >&2
      echo "LOG_METAL: ERROR — T3K topology still degraded: ERISC relay dead after recovery. Hardware needs engineer attention or host reboot." >&2
      exit 1
    fi
    raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>&1) || true
    n_chips=$(echo "$raw_output" | tr -d '\r' | grep -E '^[0-9]+$' | tail -1) || true
    if [[ -z "$n_chips" ]]; then n_chips="0"; fi
    if [[ "$n_chips" -lt 8 ]]; then
      echo "LOG_METAL: ERROR — T3K topology still degraded after recovery: ${n_chips}/8 chips visible." >&2
      echo "LOG_METAL: Hardware needs host reboot or engineer attention." >&2
      exit 1
    fi
    echo "LOG_METAL: [FIX TL/TM] topology recovered: ${n_chips}/8 chips visible after reset+warm-up."
  fi
  echo "LOG_METAL: T3K topology OK — ${n_chips}/8 chips visible."

  if [[ ${WARM_RING_TIMEOUT:-0} -eq 1 || ${TM_RING_TIMEOUT:-0} -eq 1 ]]; then
    echo "LOG_METAL: [FIX UP2] ring-sync timeout detected in pre-test warm-up — attempting one more reset+warm-up before test loop." >&2
    timeout 30 tt-smi -r || true
    local UP2_WARM_OUTPUT UP2_RING_TIMEOUT
    UP2_WARM_OUTPUT=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX UP2] pre-test-loop warm-up complete')
except Exception as e:
    print(f'[FIX UP2] WARNING: warm-up failed ({e})', file=sys.stderr)
" 2>&1 || true)
    echo "$UP2_WARM_OUTPUT"
    UP2_RING_TIMEOUT=0
    if echo "$UP2_WARM_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
      UP2_RING_TIMEOUT=1
    fi
    if [[ $UP2_RING_TIMEOUT -eq 1 ]]; then
      echo "LOG_METAL: [FIX UP3] dispatch-ERISC timeout loop detected — running final tt-smi -r to clear rescue_stuck stale state, then proceeding to tests without warm-up. (#42429)" >&2
      timeout 30 tt-smi -r || true
    else
      echo "LOG_METAL: [FIX UP2] ring-sync healthy after retry — proceeding to test loop."
    fi
  fi

  record_test() {
    local rc=$?
    if [[ $rc -eq 0 && -f /tmp/gtest_last_result.xml ]]; then
      local total_skipped
      total_skipped=$(grep -oP '(?<=skipped=")[0-9]+' /tmp/gtest_last_result.xml \
                        | awk '{s+=$1} END {print s+0}' 2>/dev/null || echo 0)
      if [[ "${total_skipped:-0}" -gt 0 ]]; then
        echo "LOG_METAL: ERROR — exit 0 but ${total_skipped} test(s) SKIPPED in GTest XML." >&2
        echo "LOG_METAL: T3K topology may have degraded mid-run — treating as failure." >&2
        rc=1
      fi
      rm -f /tmp/gtest_last_result.xml
    fi
    fail+=$rc
    if [[ $rc -ne 0 ]]; then
      echo "LOG_METAL: test returned rc=$rc — resetting hardware via tt-smi"
      timeout 30 tt-smi -r || true
      local post_warm_output post_ring_timeout
      post_warm_output=$(python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] post-reset warm-up complete — base-UMD channels cleared')
except Exception as e:
    print(f'[FIX GS-3] WARNING: post-reset warm-up failed ({e}) — next test may still skip', file=sys.stderr)
" 2>&1 || true)
      echo "$post_warm_output"
      post_ring_timeout=0
      if echo "$post_warm_output" | sed 's/\x1b\[[0-9;]*m//g; s/\r//g' | grep -qE "(FIX TK|FIX DT-1|ring_sync_already_timed_out|Timeout after [0-9]+ ms.*master chan|fabric_ring_sync_timed_out|Timeout \([0-9]+ ms\) waiting for physical cores|rescue of stuck dispatch cores|rescue_stuck_dispatch_cores)"; then
        post_ring_timeout=1
      fi
      if [[ $post_ring_timeout -eq 1 ]]; then
        consecutive_ring_timeout=$((consecutive_ring_timeout + 1))
        echo "LOG_METAL: [FIX UP] post-reset warm-up ring-sync timeout #${consecutive_ring_timeout}/3 — hardware not ready for traffic." >&2
        if [[ $consecutive_ring_timeout -ge 3 ]]; then
          echo "LOG_METAL: [FIX UP] INFRA_ERROR — ring-sync timeout on ${consecutive_ring_timeout} consecutive warm-ups. Hardware requires reboot. Aborting test run." >&2
          exit 1
        fi
      else
        consecutive_ring_timeout=0
      fi
    fi
  }

  fail=0
  consecutive_ring_timeout=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_racecondition_hunt_tests"

  # ===========================================================================
  # BATCH 1: AllGather — primary goal of this branch.
  # ===========================================================================
  echo "LOG_METAL: [BATCH 1/2] AllGather tests"

  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" ${TT_METAL_HOME}/tests/scripts/t3000/repro_ccl_cq0_hang.sh --solo ; record_test

  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl ; record_test
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl_multi_tensor ; record_test
  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 300 ./build/test/ttnn/unit_tests_ttnn_ccl_ops ; record_test

  sleep 2
  for ccl_mcq_test in \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0" \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0CQ1" \
      "MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksMultithreadCQ0"; do
      echo "LOG_METAL: running test_ccl_multi_cq_multi_device --gtest_filter=${ccl_mcq_test}"
      GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 600 ./build/test/ttnn/test_ccl_multi_cq_multi_device --gtest_filter="${ccl_mcq_test}"
      record_test
      sleep 1
  done

  tt-smi -r || true
  sleep 10
  python3 -u -c "
import sys
try:
    import ttnn
    m = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.close_mesh_device(m)
    print('[FIX GS-3] pre-GAP21 warm-up complete — base-UMD channels cleared')
except Exception as e:
    print(f'[FIX GS-3] WARNING: pre-GAP21 warm-up failed ({e})', file=sys.stderr)
" 2>&1 || true
  sleep 5
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap21_rapid_allgather_quiesce_stress.py::test_rapid_allgather_quiesce_stress ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap22_allgather_inflight_close.py::test_allgather_inflight_close ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap23_partial_mesh_quiesce_cycling.py::test_partial_mesh_quiesce_cycling ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap25_back_to_back_allgather_nosync.py::test_back_to_back_allgather_nosync ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap38_fixba_allgather_correctness_after_cleanup.py::test_gap38_fixba_allgather_correctness_after_cleanup ; record_test

  # ===========================================================================
  # BATCH 2: ETH/ERISC infrastructure regression tests.
  # ===========================================================================
  echo "LOG_METAL: [BATCH 2/2] ETH/ERISC infrastructure regression tests"

  GTEST_OUTPUT="xml:/tmp/gtest_last_result.xml" timeout 900 ./build/test/tt_metal/distributed/distributed_unit_tests \
    --gtest_filter='Gap1ThreePassEthLaunchFixture.*:LaunchGateLiveEriscFixture.*:ERISCHeartbeatFixture.*:PartialMeshQuiesceFixture.*:RelayBrokenTeardownFixture.*:Phase25RelayBrokenCascadeFixture.*:MmioPhase5RelayBrokenFixture.*:InitRouterSyncDeadRelayFixture.*:ParallelHeartbeatPollFixture.*:ChannelsNotReadyLifecycleFixture.*:RelayTimeoutToleranceFixture.*:TeardownReopenEthOrderingFixture.*:AsyncTeardownRaceFixture.*:AsyncTeardownMultiCQFixture.*:AsyncTeardownFabric2DFixture.*:AsyncTeardownFabric2DRepeatFixture.*:AsyncTeardownFabric1DQuiesceFixture.*:AsyncTeardownKillPredecessorFixture.*:FabricFirmwareInitializer.*:QuiesceStressFixture.*:PhaseWFixture.*:PhaseZFixture.*:FixAvRelayBrokenSysmemGuardFixture.*:ClusterTeardownHangRelayBrokenFixture.*:FixAyDeferredNonMmioResetFixture.*:FixAzL1BarrierSkipNoPriorFabricFixture.*:EthCoordPreservedOnAqSkipFixture.*:MmioEthCoordBeforeRelayGuardFixture.*:AsyncBuildPhaseRelayGuardFixture.*:WriteCorRelayGuardFixture.*:EthTrainingFabricEriscsFixture.*:RelayBrokenChipsCacheFixture.*:ReadCoreRelayGuardFixture.*:FwLaunchAddrForceResetFixture.*:FwLaunchAddrRescueFixture.*:FwLaunchAddrQuiesceFixture.*:TeardownNullControlPlaneFixture.*:UmdHeartbeatSkipExitFixture.*:Phase25RelayRetryFixture.*:FixE2AyProbeDeadFayTriggerFixture.*:FixM2DeadPeerEriscResetFixture.*:FixPlBarrierGuardDeadRelayFixture.*:FixQcNonMmioResetCoresSkipFixture.*:FixQbResetLoopEarlyBreakFixture.*:FixPyPzPhase25TopologyTimeoutFixture.*:FixQdDeadRouterMmioSkipFixture.*:FixQuReassertFlagsFixture.*:FixNyRelayMuxClusterGuardFixture.*:FixTbTopologyMapperUnknownAsicFixture.*:FixQvPhase4SkipFixture.*:FixRzStaleBaseUmdFlagFixture.*:FixTf2dFabricHeaderArgsGuardFixture.*:FixTgControlPlaneHostRankGuardFixture.*:FixThRelayMuxNoLinksGuardFixture.*:FixTkDegradedClusterChipFilterFixture.*:FixTlDegradedClusterBailBeforeCreateMeshesFixture.*:FixTiRingSyncFastSkipFixture.*:Gap74BaseFixtureTeardownGuardFixture.*:FixQwbStaleBaseUmdSkipGuardFixture.*:FixTg2BaseUmdPartialL1ClearFixture.*:FixPhYamlEthCoordGracefulSkip.*:Gap75SetUpNotActiveSkipGuard.*:Gap76FixSbIdleEthInitFabricGuard.*' ; record_test

  # GAP-84 runs FIRST in BATCH 2 — probes whether OPTION C leaves relays marginal before any other
  # ETH/ERISC test can mask the symptom.  Timeout = V_TOTAL_BUDGET(120) + P_BUDGET(90) + K(60) + slack.
  timeout 360 pytest -svv tests/nightly/t3000/ccl/test_gap84_optc_probe_no_marginal_state.py::test_gap84_optc_probe_no_marginal_state ; record_test

  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap24_rapid_close_reopen_cycling.py::test_rapid_close_reopen_cycling ; record_test
  timeout 180 pytest -svv tests/nightly/t3000/ccl/test_gap26_fixas_canary_timeout_graceful.py::test_gap26_fixas_canary_timeout_graceful ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap27_fixav_nonmmio_sysmem_reset.py::test_gap27_fixav_nonmmio_sysmem_reset ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap30_fixal_started_early_exit_timing.py::test_gap30_fixal_started_early_exit_timing ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap34_fixam_phase5b_skip_timing.py::test_gap34_fixam_phase5b_skip_timing ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap35_fixat_fixas_mmio_phase5_skip.py ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap36_fixav_relay_dead_per_device_skip.py::test_gap36_fixav_relay_dead_per_device_skip ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap37_fixba_started_state_nonmmio_cleanup.py::test_gap37_fixba_started_state_nonmmio_cleanup ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap39_fixns_single_topology_discovery.py::test_gap39_fixns_single_topology_discovery ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap40_fixae_flush_timeout_catch.py::test_gap40_fixae_flush_timeout_catch ; record_test

  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_phase25_force_reset_pass0_chain.py::test_phase25_force_reset_pass0_chain ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_cross_device_relay_broken_race.py::test_cross_device_relay_broken_race ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_l1_corruption_cascade.py::test_l1_corruption_cascade_broken ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_phase4_mux_timeout_recovery.py::test_phase4_mux_timeout_recovery ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_snapshot_deadline_false_positive.py::test_snapshot_deadline_false_positive ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_3pass_eth_launch_ordering.py::test_3pass_eth_launch_no_deadlock ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_partial_mesh_handshake_tolerance.py::test_partial_mesh_quiesce_nonfatal ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_router_sync_graceful_degradation.py::test_router_sync_no_sigabrt ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_channels_not_ready_flag_lifecycle.py::test_channels_not_ready_cleared_on_reinit ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_teardown_parallel_heartbeat_poll.py::test_teardown_heartbeat_poll_parallel ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap79_fixxy2_relay_broken_cleared_allgather.py::test_relay_broken_cleared_allgather_correctness ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap80_fixdt1_dispatch_erisc_timeout_warmup_allgather.py::test_gap80_fixdt1_dispatch_erisc_timeout_warmup_allgather ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap81_fixup3_dispatch_erisc_loop_warmup_skip_allgather.py::test_gap81_fixup3_dispatch_erisc_loop_warmup_skip_allgather ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap82_fixscaddr_eth_not_done_cores_allgather.py::test_gap82_fixscaddr_eth_not_done_cores_allgather ; record_test
  timeout 300 pytest -svv tests/nightly/t3000/ccl/test_gap83_fixrr_rs_mmio_rompostcode_recovered_allgather.py::test_gap83_fixrr_rs_mmio_rompostcode_recovered_allgather ; record_test

  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_racecondition_hunt_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttnn_udm_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttnn_udm_tests"
  ./build/test/ttnn/unit_tests_ttnn_udm

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttnn_udm_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tt_metal_multiprocess_tests() {
  local mpi_args="--allow-run-as-root"
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2.yaml
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/multi_host_fabric_tests
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_strict_connection_multi_process_rank_bindings.yaml  ./build/test/tt_metal/multi_host_fabric_tests
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2.yaml
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/t3k_2x2_ttswitch_rank_bindings.yaml ./build/test/tt_metal/multi_host_ttswitch_tests --gtest_filter="MeshDeviceTTSwitchFixture.*"

  # Big-Mesh 2x4 Regression tests
  # Tests are disabled for now due to ND hangs
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter="*BigMeshDualRankTest2x4*"
  #tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="*MeshWorkloadTest*"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter="*BigMeshDualRankMeshShapeSweep*"
}

run_t3000_ttnn_multiprocess_tests() {
  local mpi_args="--allow-run-as-root"

  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2

  # Big-Mesh 2x4 Regression tests
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/unit_tests_ttnn --gtest_filter="*LaunchOperation*"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/distributed/test_data_parallel_example.py
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_new_all_broadcast.py::test_all_broadcast_sharded_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_all_to_all_combine.py::test_all_to_all_combine_no_trace_submesh
  # Re-enable this test when we have more T3K availability
  # tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv "tests/nightly/t3000/ccl/test_point_to_point.py::test_point_to_point[silicon_arch_name=wormhole_b0-dtype=torch.bfloat16-shape_coords=((1, 1, 1, 16), ((0, 0), (0, 1)))-tile-mesh_device=(2, 4)-device_params={'fabric_config': <FabricConfig.FABRIC_1D: 1>}]"
}

run_t3000_ttnn_multiprocess_slow_tests() {
  local mpi_args="--allow-run-as-root"
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"

  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/distributed/test_submesh_not_spanning_all_ranks_T3000.py
}

run_t3000_grok_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_grok_tests"

  pytest models/experimental/grok/tests/test_grok_rms_norm.py ; fail+=$?
  pytest models/experimental/grok/tests/test_grok_attention.py ; fail+=$?
  pytest models/experimental/grok/tests/test_grok_mlp.py --timeout=500; fail+=$?
  pytest models/experimental/grok/tests/test_grok_moe.py --timeout=600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_grok_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwen3_vl_unit_tests() {
  # install qwen3_vl requirements
  uv pip install -r models/demos/qwen3_vl/requirements.txt

  # export PYTEST_ADDOPTS for concise pytest output
  export PYTEST_ADDOPTS="--tb=short"

  qwen3_vl_32b=Qwen/Qwen3-VL-32B-Instruct
  tt_cache_32b=$TT_CACHE_HOME/$qwen3_vl_32b

  # run unit tests
  MESH_DEVICE=T3K HF_MODEL=$qwen3_vl_32b TT_CACHE_PATH=$tt_cache_32b pytest models/demos/qwen3_vl/tests/ --ignore=models/demos/qwen3_vl/tests/test_ci_dispatch.py --ignore=models/demos/qwen3_vl/tests/conftest.py
}

run_t3000_deepseek_tests() {
  uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt

  export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked
  export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI
  MESH_DEVICE=T3K pytest models/demos/deepseek_v3/tests/unit --timeout 60 --durations=0
}

run_t3000_ccl_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ccl_tests"

  # all gather: 1 ring, 1 line, 1 2d, 1 sharded should be covered
  # width sharded to interleaved case using linear - using i2s_shape0 which is perf with fabric_linear
  pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_sharded_to_interleaved[wormhole_b0-fabric_linear-i2s_shape0-perf-1-Layout.TILE-DataType.BFLOAT16-mesh_device0]
  # 10 iteration trace test with fabric ring (dit_shape now in test_ttnn_all_gather, no barrier parameters)
  pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_ttnn_all_gather[wormhole_b0-fabric_ring-mem_config_input0-mem_config_ag0-dit_shape-perf-1link-mesh_device0]
  # 2D fabric case – hanging on main? tracking with issue #30250
  # pytest tests/nightly/t3000/2d_ccl/test_minimal_all_gather_async.py::test_all_gather_async_training_shapes[wormhole_b0-fabric_2d_dynamic_linear-check-mem_config_input0-mem_config_ag0-tt_training_test_one-mesh_device0-1link]
  # training shapes - Re-enable this test when we have more T3K availability
  # pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_training_shapes[wormhole_b0-fabric_linear-mem_config_input0-mem_config_ag0-tt_training_test_four-check-mesh_device0-1link]

  # reduce scatter: 1 ring, 1 line, 1 2d, 1 sharded should be covered
  # sharded intermediate case with cluster axis 1
  pytest tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_minimal_async_linear_sharded
  # composite case
  pytest tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_training_shapes[wormhole_b0-fabric_linear-random-mem_config_input0-mem_config_rs0-tt_training_test_one-check-mesh_device0-1link]
  # long trace test on dim=1 with ring, currently hanging when run in the suite even though it passes when run in isolation - Re-enable this test when we have more T3K availability
  # pytest tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async[wormhole_b0-fabric_ring-random-mem_config_input0-mem_config_rs0-scatter_dim_1_test_one-perf-no_barrier_with_persistent-1link-mesh_device0]
  # long running dim = 3 trace test without barrier and with persistent buffers
  pytest tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async[wormhole_b0-fabric_ring-random-mem_config_input0-mem_config_rs0-padded_dim_2_test_two-perf-no_barrier_with_persistent-1link-mesh_device0]

  # all reduce: 1 test should be enough
  # 4 chip test with bfloat8_b
  pytest tests/nightly/t3000/ccl/test_all_reduce.py::test_ring_all_reduce_post_commit -k "2x4x2048x32-bfloat8_b-DRAM-4-1"

  # p2p: 1 test should be enough
  # trace test with device delay
  pytest tests/nightly/t3000/ccl/test_point_to_point.py::test_point_to_point_with_device_delay -k tile
  pytest tests/ttnn/unit_tests/operations/debug/test_generic_op.py::test_point_to_point

  # all broadcast: row major + tile test
  # both rm and tile test are called here
  pytest tests/nightly/t3000/ccl/test_new_all_broadcast.py::test_all_broadcast_trace

  # all to all dispatch: 1 test for 2d and 1 for 1d linear should be enough
  # fabric 1d linear test on cluster axis 0 as other CCL tests aren't testing on this axis
  pytest tests/nightly/t3000/ccl/test_all_to_all_dispatch.py::test_all_to_all_dispatch_trace[wormhole_b0-DataType.BFLOAT16-MAX_LINKS-dram-dram-s128-7168-8-8-8-cluster_axis_0-2x4_grid-True-fabric_1d_linear]
  # fabric 2d test on cluster axis 1
  pytest tests/nightly/t3000/ccl/test_all_to_all_dispatch.py::test_all_to_all_dispatch_no_trace[wormhole_b0-DataType.BFLOAT16-MAX_LINKS-b1s3-l1-7168-8-8-cluster_col-2x4_grid-False-fabric_2d]

  # all to all combine: 1 test for 1d ring and 1 for 2d should be enough
  pytest tests/nightly/t3000/ccl/test_all_to_all_combine.py::test_all_to_all_combine_no_trace[wormhole_b0-DataType.BFLOAT16-None-dram-dram-2-random-True-2-7000-8-8-8-fabric_1d_ring_axis_1]
  # fabric 2d test on cluster axis 0 - Re-enable this test when we have more T3K availability
  # pytest tests/nightly/t3000/ccl/test_all_to_all_combine.py::test_all_to_all_combine_no_trace[wormhole_b0-DataType.BFLOAT16-None-dram-dram-2-random-True-2-7000-8-8-8-fabric_2d_axis_0]

  # neighbor pad: 1D correctness check + 2D correctness check
  pytest tests/nightly/t3000/ccl/test_neighbor_pad_async.py::test_neighbor_pad_async_1d -k "zeros_width_dim-check"
  pytest tests/nightly/t3000/ccl/test_neighbor_pad_async.py::test_neighbor_pad_async_2d -k "small_5d_h0w1"

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ccl_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tt_dit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tt_dit_tests"

  #Timestep Encoding (Currently used in Wan2.2)
  pytest models/tt_dit/tests/unit/test_embeddings.py::test_timestep_encoding ; fail+=$?

  #Wan2.2 Time Text Image Embedding
  pytest models/tt_dit/tests/unit/test_embeddings.py::test_wan_time_text_image_embedding  -k "t3k" ; fail+=$?

  #T5 Encoder
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/encoders/t5/test_t5_full.py::test_t5_encoder -k "t3k" ; fail+=$?

  #UMT5 Encoder
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/encoders/umt5/test_umt5.py -k "t3k" ; fail+=$?

  #Clip Encoder
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/encoders/clip/test_clip_full_projection.py -k 1x4-t3k ; fail+=$?

  #Image DiTs VAE with one iteration pcc and perf test
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/models/sd35/test_vae_sd35.py::test_sd35_vae_vae_decoder -k "t3k" ; fail+=$?

  #Flux1 Single Transformer Block and other Image DiTs Transformer blocks
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/models/flux1/test_transformer_flux1.py::test_transformer -k 2x4sp0tp1 ; fail+=$?

  #DITs Wan2.2 VAE
  pytest models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder[wormhole_b0-device_params0-2x4_h1_w0-bf16-chunk_1-check_output-fake_weights-0-1-_1f-480p] ; fail+=$?

  #DITs Wan2.2 Transformer
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/models/wan2_2/test_transformer_wan.py::test_wan_transformer_model[wormhole_b0-short_seq-2x4sp0tp1-True] ; fail+=$?

  #Mochi Transformer
  DIT_UNIT_TEST=1 pytest models/tt_dit/tests/models/mochi/test_transformer_mochi.py::test_mochi_transformer_model[wormhole_b0-device_params0-no_load_cache-no_test_attention_mask-short_seq-2x4sp0tp1-True] ; fail+=$?

  #Mochi VAE main component
  FAKE_DEVICE=T3K pytest models/tt_dit/tests/models/mochi/test_vae_mochi.py::test_tt_resblock_forward[wormhole_b0-mesh_device0-device_params0-1link-l768] ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_tt_dit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_t3000_mistral-small-3.1-24b-vision_unit_tests() {
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mistral-small-3.1-24b-vision_unit_tests"

  mistral24b=mistralai/Mistral-Small-3.1-24B-Instruct-2503
  tt_cache_mistral24b=$TT_CACHE_HOME/$mistral24b

  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_conv2d.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_vision_rms.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_vision_mlp.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_vision_attention.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_pixtral_transformer.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_patch_rot_emb.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_vision_model.py ; fail+=$?
  HF_MODEL=$mistral24b TT_CACHE_PATH=$tt_cache_mistral24b pytest --timeout 600 models/tt_transformers/tests/multimodal/mistral_24b/test_vision_tower.py ; fail+=$?

  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mistral-small-3.1-24b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tttv2_fast_unit_tests() {
  fail=0

  # Run non-module models/common unit tests
  pytest --tb=short --ignore=models/common/tests/modules models/common/tests ; fail+=$?

  # [INFO] HF_MODEL Only used for test_*_1d_vs_reference_from_model_args, which will retire with TTTv1
  # Run MLP1D fast unit tests (full set is run in t3k_e2e_tests.yaml to match timeout values and frequency of runs)
  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/mlp_1d \
  pytest models/common/tests/modules/mlp/test_mlp_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.mlp.mlp_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run RMSNorm1D tests
  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/rmsnorm_1d \
  pytest models/common/tests/modules/rmsnorm/test_rmsnorm_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.rmsnorm.rmsnorm_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run Rope1D tests
  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/rope_1d \
  pytest models/common/tests/modules/rope/test_rope_1d.py \
    -m "not slow" \
    --durations=10 \
    --tb=short \
    --cov=models.common.modules.rope.rope_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run LMHead1D tests
  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/lm_head_1d \
  pytest models/common/tests/modules/lm_head/test_lm_head_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.lm_head.lm_head_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run Attention1D tests
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/attention_1d \
  pytest models/common/tests/modules/attention/test_attention_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.attention.attention_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run Embedding1D tests
  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/embedding_1d \
  pytest models/common/tests/modules/embedding/test_embedding_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.embedding.embedding_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run Penalties1D tests
  pytest models/common/tests/modules/sampling/test_penalties_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.sampling.penalties_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  # Run Sampling1D tests
  pytest models/common/tests/modules/sampling/test_sampling_1d.py \
    -m "not slow" \
    --tb=short \
    --durations=10 \
    --cov=models.common.modules.sampling.sampling_1d \
    --cov-report=term-missing \
    --cov-config=models/common/tests/setup.cfg ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run ttmetal tests
  run_t3000_ttmetal_tests

  # Run ttfabric tests
  run_t3000_ttfabric_tests

  # Run ttnn tests
  run_t3000_ttnn_tests

  # Run grok tests
  run_t3000_grok_tests

  # Run tt_dit tests
  run_t3000_tt_dit_tests

  # Run tttv2 fast unit tests
  run_t3000_tttv2_fast_unit_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
