#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""Own paired B32/B16 or candidate-only B32 lifecycles in one goal12 queue reservation.

No profiler or allocator tracking. Invoke through device_queue/run.py. Startup
and cleanup count against the device-hour box; B8 is never launched here.
"""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
REL = Path('models/autoports/qwen_qwen3_8_27b')
TESTS = REPO / REL / 'tests'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_port_available(host='127.0.0.1', port=8000):
    # Match asyncio.start_server: reuse eligible TIME_WAIT, never a live listener.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))


def checked_run(command, deadline, log_path, cleanup_reserve=90):
    remaining = deadline-time.monotonic()-cleanup_reserve
    if remaining <= 0:
        raise TimeoutError('Device-hour box has only cleanup reserve remaining')
    with log_path.open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       timeout=remaining, check=True)


def candidate_reference(path, expected_block=4096):
    """Validate retained native baseline before reserving any serving lifecycle."""
    if path is None:
        raise ValueError('Candidate-only mode requires the retained B32 --reference')
    path = path.resolve()
    baseline = json.loads(path.read_text())
    identity = baseline['server_identity']
    expected_revision = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    if (baseline.get('batch') != 32 or baseline.get('concurrency') != 10
            or baseline.get('output_cap') != 81920 or baseline.get('request_timeout') != 12000
            or identity.get('layer_count') != 64 or identity.get('max_num_seqs') != 32
            or identity.get('model_revision') != expected_revision
            or identity.get('tokenizer_revision') != expected_revision):
        raise ValueError('Retained baseline has incompatible model, cohort or release limits')
    argv = identity['server_command']
    for option in ('--block-size', '--mamba-block-size'):
        if argv[argv.index(option)+1] != str(expected_block):
            raise ValueError('Retained baseline cache blocks differ from required reference')
    if expected_block == 2048 and not any('num_gpu_blocks="353"' in line for line in baseline.get('metrics_after', [])):
        raise ValueError('Retained pool baseline must prove353 allocated IDs in vLLM metrics')
    spec = importlib.util.spec_from_file_location('cache_candidate_probe', TESTS/'agentic_batch_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    if baseline.get('cohort_sha256') != probe.sha(probe.make_cohort()):
        raise ValueError('Current native cohort differs from retained baseline')
    if not baseline.get('invariant_evidence_complete') or not baseline.get('all_native_token_ids_present'):
        raise ValueError('Retained baseline native tokens or trace/cache invariants are incomplete')
    if not probe.compare_pair(baseline, baseline)['exact']:
        raise ValueError('Retained baseline needs20 complete native responses across two rounds')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cohort_sha256=baseline['cohort_sha256'], model_revision=expected_revision,
        source_commits=identity.get('source_commits'), reference_scheduler_block_size=expected_block, default_scheduler_block_size=4096,
        rounds_seconds=[r['elapsed_seconds'] for r in baseline['rounds']],
        basis='Retained instrumented paired-02 B32 previous run; candidate is a fresh process. No fresh baseline or B16 launch.',
        timing_limit='Coordinator must assess cache warmth/timing noise and request a warm-baseline repeat only if needed.',
        source_limit='Observed baseline source commits do not establish all live overlay bytes. Candidate source hashes and exact request/token comparison supplement this receipt; no source-equivalence claim from commit ID alone.')


def main(args):
    candidate_pool = getattr(args, 'candidate_pool_blocks', None)
    if candidate_pool not in (None,385):
        raise ValueError('Pool candidate must be385')
    candidate_block = 2048 if candidate_pool else getattr(args, 'candidate_block_size', None)
    if candidate_block not in (None, 2048):
        raise ValueError('Candidate-only block must be2048')
    baseline = candidate_reference(getattr(args, 'reference', None), expected_block=2048 if candidate_pool else 4096) if candidate_block else None
    if not candidate_block and getattr(args, 'reference', None):
        raise ValueError('--reference is only valid with --candidate-block-size2048')
    batches = [32] if candidate_block else [32, 16]
    root = args.output.resolve()
    root.relative_to(REPO)  # All paths must be translatable into the shared container.
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh arm directory; refusing to reuse ownership state')
    root.mkdir(parents=True, exist_ok=True)
    if candidate_pool and args.box_hours > .3:
        raise ValueError('Preliminary pool gate box is at most.3 mesh-hour')
    if args.box_hours <= 0 or args.box_hours > 1:
        raise ValueError('Coordinator effort box is positive and at most1device-hour')
    spec = importlib.util.spec_from_file_location('batch_owned_server', TESTS/'benchmark_server.py')
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    started = time.monotonic()
    deadline = started+args.box_hours*3600
    relay = None
    active_state = None
    errors = []
    arms = []
    result = dict(owner='cache_pool_expansion' if candidate_pool else 'cache_block_rounding' if candidate_block else 'batch-fit', started_unix=time.time(), box_hours=args.box_hours,
                  batches=batches, scheduler_block_size=candidate_block or 4096, scored=False, arms=arms,
                  retained_baseline=baseline, num_gpu_blocks_override=candidate_pool, output_cap=81920, request_timeout=12000)
    if candidate_block:
        result['candidate_source_sha256'] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [TESTS/'agentic_batch_driver.py', TESTS/'agentic_batch_probe.py',
                         TESTS/'benchmark_server.py', REPO/REL/'tt/cache_block_config.py',
                         REPO/REL/'tt/vllm_cache_specs.py', REPO/REL/'tt/generator_vllm.py']}
        write(root/'retained-baseline.json', baseline)
    def interrupted(signum, frame):
        raise InterruptedError(f'Coordinator interrupted driver with signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        check_port_available()
        with (root/'relay.log').open('w') as log:
            relay = subprocess.Popen([sys.executable, str(TESTS/'benchmark_proxy.py'),
                                      str(root/'relay-owner.json')], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(50):
            if relay.poll() is not None:
                raise RuntimeError('Owned relay exited before readiness')
            if (root/'relay-owner.json').exists():
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Owned relay readiness missing')
        for batch in batches:
            arm = root/f'b{batch}'
            observed = arm/'observed'
            observed.mkdir(parents=True)
            active_state = arm/'owned_server.json'
            server_config = observed/'server.json'
            events = observed/'events.jsonl'
            launch_start = time.monotonic()
            record = dict(batch=batch, status='starting', started_unix=time.time(),
                          server_config=str(server_config), events=str(events))
            arms.append(record)
            write(root/'summary.json', result)
            try:
                checked_run([sys.executable, str(TESTS/'benchmark_server.py'), '--max-num-seqs', str(batch),
                             '--output', str(server_config), '--serving-events', str(events),
                             '--shutdown-timeout', '30', '--scheduler-block-size', str(candidate_block or 4096)] +
                            (['--num-gpu-blocks-override', str(candidate_pool)] if candidate_pool else []),
                            deadline, observed/'launch.log')
                record['startup_seconds'] = time.monotonic()-launch_start
                if candidate_block:
                    identity = json.loads(server_config.read_text())
                    if (identity.get('model_revision') != baseline['model_revision']
                            or identity.get('tokenizer_revision') != baseline['model_revision']
                            or identity.get('layer_count') != 64 or identity.get('max_num_seqs') != 32):
                        raise ValueError('Candidate observed identity differs from retained model/checkpoint')
                    for option in ('--block-size', '--mamba-block-size'):
                        argv = identity['server_command']
                        if argv[argv.index(option)+1] != str(candidate_block):
                            raise ValueError('Candidate observed launch identity has incorrect cache blocks')
                if candidate_pool and argv[argv.index('--num-gpu-blocks-override')+1] != str(candidate_pool):
                    raise ValueError('Observed pool override differs')
                command = [sys.executable, str(TESTS/'agentic_batch_probe.py'), '--batch', str(batch),
                           '--server-config', str(server_config), '--events', str(events),
                           '--output', str(arm/'probe'), '--output-cap', '81920', '--request-timeout', '12000']
                if candidate_block:
                    command += ['--reference', baseline['path']]
                elif batch == 16:
                    command += ['--reference', str(root/'b32/probe/result.json')]
                record['status'] = 'probing'
                write(root/'summary.json', result)
                checked_run(command, deadline, observed/'probe.log')
                probe = json.loads((arm/'probe/result.json').read_text())
                record.update(status=probe['status'], measured_wall_seconds=probe.get('measured_wall_seconds'),
                              invariants_complete=probe.get('invariant_evidence_complete'),
                              exact_pair=probe.get('pair', {}).get('exact'))
                if not probe.get('invariant_evidence_complete') or not probe.get('all_native_token_ids_present'):
                    raise RuntimeError('Probe trace/cache/native-token evidence incomplete; no candidate acceptance')
            finally:
                if active_state.exists():
                    server.helper('inspect', active_state)
                    server.helper('stop', active_state)
                record['reservation_seconds'] = time.monotonic()-launch_start
                write(root/'summary.json', result)
                active_state = None
        result['status'] = 'review_required'
        result['acceptance'] = 'No automatic selection; coordinator reviews exactness, latency, occupancy and cache evidence'
        if candidate_pool:
            result['candidate_confirmation'] = dict(pool_ids=385, exact_native_tokens=True,
                postwarmup_captures=0, host_logits_reads=0, status='preliminary_only_pending_pressure_confirmation',
                limitation='No capacity benefit or release selection from this short cohort')
        elif candidate_block:
            result['candidate_confirmation'] = dict(additional_prefix_hits_per_return_min=2048,
                previous_sent_hit_fraction_min=0.98, exact_native_tokens=True, return_turn_time='lower',
                postwarmup_captures=0, preemptions=0, status='pending_coordinator_review')
    except BaseException as error:
        errors.append(repr(error))
        result['status'] = 'incomplete'
        result['missing_reason'] = repr(error)
        if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            result['cutoff_policy'] = 'Effort box reached; partial probe is unscored and gives no speedup or failure score. Coordinator must re-estimate before rerun.'
    finally:
        if active_state is not None and active_state.exists():
            try:
                server.helper('inspect', active_state)
                server.helper('stop', active_state)
            except BaseException as error:
                errors.append('server cleanup: '+repr(error))
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:
                relay.wait(timeout=10)
            except subprocess.TimeoutExpired:
                relay.kill()
                relay.wait()
        result.update(elapsed_seconds=time.monotonic()-started,
                      used_device_hours=(time.monotonic()-started)/3600, errors=errors,
                      relay_exit=None if relay is None else relay.returncode,
                      owned_server_states_remaining=[str(p) for p in root.glob('b*/owned_server.json')])
        write(root/'summary.json', result)
    return 0 if result.get('status') == 'review_required' and not errors else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-pool-blocks', type=int, choices=[385], help='Preliminary exact B32/block2048 pool gate against retained353-ID baseline')
    parser.add_argument('--candidate-block-size', type=int, choices=[2048], help='Run only B32 candidate against retained reference; no baseline/B16 launch')
    parser.add_argument('--reference', type=Path, help='Retained paired-02/b32/probe/result.json')
    parser.add_argument('--box-hours', type=float, default=1.0)
    raise SystemExit(main(parser.parse_args()))
