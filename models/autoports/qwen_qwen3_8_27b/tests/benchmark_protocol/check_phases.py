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
        if self.inputs.prompt_lens is not None:
            fwd = self._forward_with_model_input(self.inputs)
            return self._finish_front_packed_sync_impl(None, fwd=fwd)
        return self.controller.submit_async_decode(self.inputs, steady_decode_fast_path=True)

    def _forward_with_model_input(self, inputs):
        return NS(model_input=inputs)

    def _finish_front_packed_sync_impl(self, grammar_output, *, fwd):
        return NS(fwd=fwd)


class Controller:
    def submit_async_decode(self, model_input, *, steady_decode_fast_path):
        return Output(self, model_input)

    def complete_decode_step(self, submission, model_input, context, scheduled_rows=None):
        assert model_input is submission._model_input
        assert context is submission._context
        return NS(completion_time_ns=time.perf_counter_ns())


class Output:
    def __init__(self, controller, model_input):
        self._controller = controller
        self._context = NS(submit_time_ns=time.perf_counter_ns())
        self._model_input = model_input

    def _get_output_impl(self):
        completed = self._controller.complete_decode_step(submission=self,
            model_input=self._model_input, context=self._context, scheduled_rows=None)
        self.output_built_at = time.perf_counter()
        return completed


class Scheduler:
    def schedule(self, throttle_prefills=False):
        self.inside_schedule = time.perf_counter()
        return NS(num_scheduled_tokens=dict(self.tokens))

    def update_from_output(self, scheduler_output, model_runner_output):
        self.inside_consume = time.perf_counter()
        return {'consumed': list(scheduler_output.num_scheduled_tokens)}


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
    scheduler.tokens = {'a': 4096, 'b': 4096}
    runner.inputs = NS(row_req_ids=['a', 'b'], input_positions=Array([0, 0]),
                       prompt_lens=[4096, 4096], intermediate_prefill_mask=[False, False])
    prefill_schedule = scheduler.schedule(throttle_prefills=False)
    first_schedule_inside = scheduler.inside_schedule
    prefill_output = runner.execute_model(prefill_schedule)
    scheduler.update_from_output(prefill_schedule, prefill_output)
    scheduler.tokens = {'a': 1, 'b': 1}
    runner.inputs = NS(row_req_ids=['a', 'b'], input_positions=Array([4096, 4096]), prompt_lens=None)
    first_schedule = scheduler.schedule()
    first = runner.execute_model(first_schedule)
    # Second step is already submitted, with reordered rows and stale host positions.
    runner.inputs = NS(row_req_ids=['b', 'a'], input_positions=Array([4096, 4096]), prompt_lens=None)
    second_schedule = scheduler.schedule()
    second = runner.execute_model(second_schedule)
    for submission, schedule_output in ((first, first_schedule), (second, second_schedule)):
        completed_output = submission._get_output_impl()
        scheduler.update_from_output(scheduler_output=schedule_output, model_runner_output=completed_output)
    signal.getsignal(signal.SIGUSR1)()
    events = [json.loads(line) for line in output.read_text().splitlines()]
    phases = [event for event in events if event['event'] == 'phase']
    assert len(phases) == 3
    assert phases[0]['phase'] == 'prefill'
    assert [row['position'] for row in phases[1]['rows']] == [4096, 4096]
    assert [row['position'] for row in phases[2]['rows']] == [4097, 4097]
    assert [row['request'] for row in phases[2]['rows']] == ['b', 'a']
    assert [phase['step'] for phase in phases] == [1, 2, 3]
    assert all(phase['end'] >= phase['start'] for phase in phases)
    assert all(phase['complete_ns'] >= phase['submit_ns'] for phase in phases[1:])
    assert phases[1]['end'] >= first.output_built_at
    assert phases[2]['end'] >= second.output_built_at
    assert phases[0]['start'] <= first_schedule_inside
    consumed = [event for event in events if event['event'] == 'consumed']
    assert [event['step'] for event in consumed] == [1, 2, 3]
    assert all(event['end'] >= phase['end'] for event, phase in zip(consumed, phases))
    assert consumed[-1]['end'] >= scheduler.inside_consume
    flush = json.loads(output.with_suffix('.flush.json').read_text())
    assert flush['pending_records'] == 0
    print(json.dumps({'status': 'pass', 'scope': 'host mocks; no hardware/runtime performance claim',
        'checks': ['sync forward/finalization signature', 'async keyword call signature',
                   'immutable context mapping across overlapping submissions',
                   'request identity after row reorder', 'resident-feedback position advancement',
                   'output construction included after readback',
                   'schedule start retained through execute',
                   'scheduler consumption matched across overlapping submissions',
                   'pending records zero at idle flush',
                   'idle buffer flush', '64-layer ready identity'],
        'phases': len(phases), 'requests': len([e for e in events if e['event'] == 'request'])}, indent=2))
signal.signal(signal.SIGUSR1, previous_handler)
