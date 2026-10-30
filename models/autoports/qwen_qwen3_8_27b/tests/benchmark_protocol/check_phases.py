"""Host mocks for benchmark phase hook signatures and asynchronous identities."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types

NS = types.SimpleNamespace


class Array(list):
    def tolist(self):
        return list(self)


class Runner:
    def execute_model(self, scheduler_output):
        if getattr(self, 'drain', False):
            self.controller.wait_for_all_pending_async_steps()
        self.build_model_input(scheduler_output, None)
        if self.inputs.prompt_lens is not None:
            fwd = self._forward_with_model_input(self.inputs)
            return self._finish_front_packed_sync_impl(None, fwd=fwd)
        return self.controller.submit_async_decode(self.inputs, steady_decode_fast_path=True)

    def _forward_with_model_input(self, inputs):
        return NS(model_input=inputs)

    def _finish_front_packed_sync_impl(self, grammar_output, *, fwd):
        completed = NS(context=NS(req_ids=list(fwd.model_input.row_req_ids)))
        self.controller.enqueue_completed_decode_step(completed)
        return NS(fwd=fwd)

    def build_model_input(self, scheduler_output, grammar_output):
        for completed in list(self.controller.completed):
            self.controller.apply_completed_decode_step(completed,
                suppress_output_req_ids=set(), skip_state_req_ids=set())
            self.controller.completed.remove(completed)
        return self.inputs


class Controller:
    def __init__(self):
        self.completed = []
        self.outstanding = []

    def submit_async_decode(self, model_input, *, steady_decode_fast_path):
        output = Output(self, model_input)
        self.outstanding.append(output)
        return output

    def complete_decode_step(self, submission, model_input, context, scheduled_rows=None):
        assert model_input is submission._model_input
        assert context is submission._context
        return NS(completion_time_ns=time.perf_counter_ns(), context=context)

    def enqueue_completed_decode_step(self, completed):
        self.completed.append(completed)

    def apply_completed_decode_step(self, completed, *, suppress_output_req_ids=None, skip_state_req_ids=None):
        pass

    def wait_for_all_pending_async_steps(self):
        for output in self.outstanding:
            output.ensure_finalized()


class Output:
    def __init__(self, controller, model_input):
        self._controller = controller
        self._context = NS(submit_time_ns=time.perf_counter_ns(), req_ids=list(model_input.row_req_ids))
        self._model_input = model_input
        self._finalized = False

    def ensure_finalized(self):
        if not self._finalized:
            self._result = self._get_output_impl()
            self._finalized = True
        return self._result

    def _get_output_impl(self):
        completed = self._controller.complete_decode_step(submission=self,
            model_input=self._model_input, context=self._context, scheduled_rows=None)
        self.output_built_at = time.perf_counter()
        self._controller.enqueue_completed_decode_step(completed)
        return completed


class Scheduler:
    def __init__(self):
        self.running = []
        self.waiting = []
        self.finish = []

    def add_request(self, request):
        self.running.append(request)

    def schedule(self, throttle_prefills=False):
        self.inside_schedule = time.perf_counter()
        return NS(num_scheduled_tokens=dict(self.tokens), finished_req_ids=[])

    def update_from_output(self, scheduler_output, model_runner_output):
        self.inside_consume = time.perf_counter()
        self.running = [r for r in self.running if r.request_id not in self.finish]
        return {0: NS(outputs=[NS(request_id=rid, new_token_ids=[1],
                                  finish_reason='length' if rid in self.finish else None)
                               for rid in scheduler_output.num_scheduled_tokens],
                      finished_requests=set(self.finish))}


sys.modules['vllm_tt_plugin.model_runner'] = NS(TTModelRunner=Runner)
sys.modules['vllm_tt_plugin.async_decode'] = NS(TTAsyncDecodeController=Controller, AsyncTTModelRunnerOutput=Output)
sys.modules['vllm_tt_plugin.scheduler'] = NS(TTScheduler=Scheduler)
model_dir = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('benchmark_phases_under_test', model_dir / 'tt/benchmark_phases.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
previous_handler = signal.getsignal(signal.SIGUSR1)
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'phases.jsonl'
    os.environ['QWEN38_BENCHMARK_PHASES'] = str(output)
    module.install(NS(batch_size=8, model=NS(layer_indices=list(range(64)), precision_config={'selected': 'test'})))
    runner = Runner()
    runner.requests = {'a': NS(prompt_token_ids=[1] * 4096), 'b': NS(prompt_token_ids=[2] * 4096)}
    runner.input_batch = NS(req_ids=['a', 'b'], num_reqs=2)
    runner.controller = Controller()
    runner.controller.runner = runner
    scheduler = Scheduler()
    scheduler.add_request(NS(request_id='a'))
    scheduler.add_request(NS(request_id='b'))
    scheduler.tokens = {'a': 4096, 'b': 4096}
    runner.inputs = NS(row_req_ids=['a', 'b'], input_positions=Array([0, 0]),
                       prompt_lens=[4096, 4096], intermediate_prefill_mask=[False, False])
    prefill_schedule = scheduler.schedule(throttle_prefills=False)
    first_schedule_inside = scheduler.inside_schedule
    prefill_output = runner.execute_model(prefill_schedule)
    scheduler.tokens = {'a': 1, 'b': 1}
    runner.inputs = NS(row_req_ids=['a', 'b'], input_positions=Array([4096, 4096]), prompt_lens=None)
    first_schedule = scheduler.schedule()
    first = runner.execute_model(first_schedule)
    # Old prefill consumption happens only after the next decode was submitted.
    scheduler.update_from_output(prefill_schedule, prefill_output)
    # Second step is already submitted, with reordered rows and stale host positions.
    runner.inputs = NS(row_req_ids=['b', 'a'], input_positions=Array([4096, 4096]), prompt_lens=None)
    second_schedule = scheduler.schedule()
    second = runner.execute_model(second_schedule)
    # New prefill naturally drains the two older decodes; their state is applied
    # inside its build, while their scheduler consumption happens afterward.
    scheduler.add_request(NS(request_id='c'))
    runner.requests['c'] = NS(prompt_token_ids=[3] * 4096)
    runner.inputs = NS(row_req_ids=['c'], input_positions=Array([0]),
                       prompt_lens=[4096], intermediate_prefill_mask=[False])
    scheduler.tokens = {'c': 4096}
    next_prefill_schedule = scheduler.schedule()
    runner.drain = True
    next_prefill_output = runner.execute_model(next_prefill_schedule)
    for submission, schedule_output in ((first, first_schedule), (second, second_schedule)):
        completed_output = submission.ensure_finalized()
        scheduler.finish = ['a', 'b'] if submission is second else []
        scheduler.update_from_output(scheduler_output=schedule_output, model_runner_output=completed_output)
    scheduler.finish = ['c']
    scheduler.update_from_output(next_prefill_schedule, next_prefill_output)
    signal.getsignal(signal.SIGUSR1)()
    events = [json.loads(line) for line in output.read_text().splitlines()]
    phases = [event for event in events if event['event'] == 'phase']
    assert len(phases) == 4
    assert phases[0]['phase'] == 'prefill'
    assert [row['position'] for row in phases[1]['rows']] == [4096, 4096]
    assert [row['position'] for row in phases[2]['rows']] == [4097, 4097]
    assert [row['request'] for row in phases[2]['rows']] == ['b', 'a']
    assert [phase['step'] for phase in phases] == [1, 2, 3, 4]
    assert all(phase['end'] >= phase['start'] for phase in phases)
    assert all(phase['complete_ns'] >= phase['submit_ns'] for phase in phases[1:3])
    assert phases[1]['end'] >= first.output_built_at
    assert phases[2]['end'] >= second.output_built_at
    assert phases[0]['start'] <= first_schedule_inside
    consumed = [event for event in events if event['event'] == 'consumed']
    assert [event['step'] for event in consumed] == [1, 2, 3, 4]
    assert all(event['end'] >= phase['end'] for event, phase in zip(consumed, phases))
    assert consumed[-1]['end'] >= scheduler.inside_consume
    flush = json.loads(output.with_suffix('.flush.json').read_text())
    assert flush['pending_records'] == 0
    applies = [event for event in events if event['event'] == 'state_apply']
    assert [(e['step'], e['phase'], e['enclosing_step']) for e in applies] == [
        (1, 'prefill', 2), (2, 'decode', 4), (3, 'decode', 4)]
    drains = [event for event in events if event['event'] == 'drain']
    assert drains[0]['origin_steps'] == [2, 3] and drains[0]['step'] == 4
    assert drains[0]['origin_phases'] == ['decode', 'decode']
    builds = {e['step']: e for e in events if e['event'] == 'build_input'}
    assert drains[0]['end'] <= builds[4]['start']
    assert consumed[-2]['finished'] == ['a', 'b'] and consumed[-1]['finished'] == ['c']
    assert len([e for e in events if e['event'] == 'admitted']) == 3
    print(json.dumps({'status': 'pass', 'scope': 'host mocks; no hardware/runtime performance claim',
        'checks': ['sync forward/finalization signature', 'async keyword call signature',
                   'immutable context mapping across overlapping submissions',
                   'request identity after row reorder', 'resident-feedback position advancement',
                   'output construction included after readback',
                   'schedule start retained through execute',
                   'scheduler consumption matched across overlapping submissions',
                   'pending records zero at idle flush',
                   'prefill deferred state remains prefill inside later decode build',
                   'decode drain and state apply remain decode inside next prefill',
                   'drain-driven resolution and later repeated finalization emit once',
                   'admission and exact consumed-finish request lifetimes',
                   'idle buffer flush', '64-layer ready identity'],
        'phases': len(phases), 'requests': len([e for e in events if e['event'] == 'request'])}, indent=2))
signal.signal(signal.SIGUSR1, previous_handler)
