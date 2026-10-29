# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The multi-user adapter under the real vLLM + vllm-tt-plugin host stack, on CPU: a real VllmConfig resolved by the
plugin's platform hook (the class's capabilities under the plugin it runs on), the real TTScheduler with its KV cache
manager (block tables, preemption under KV pressure, the TT chunk core, decode interleave), the real TTModelRunner
(persistent batch, state-slot allocation and remaps, prompt lens / start positions / final masks, reload plans, host
sampling, async decode submission and readback), driven like vLLM's EngineCore: step() for synchronous scheduling and
the two-deep batch queue of step_with_batch_queue for async scheduling. Only the device is fake: the rows-mode model
and PP prefill of test_vllm_rows_fake.py, which expose the contract's API and check every step (positions against the
slot's history, KV through the page tables, conv history clobbered by non-preserve launches, one PP prompt).

Each completed request's tokens must equal the reference: greedy = the fake's next-token rule over prompt + outputs;
sampled (device sampling, seeded) = the rule mixed with the request's seed and position, the first token host-sampled
greedy from the prefill logits (sharply peaked). Runs: max_num_seqs 1 and 8 (and 16 = two launches), synchronous and
async scheduling, host and device sampling, mixed prompt lengths (long ones are chunked while others decode),
aborts, and a small KV pool that forces preemptions.

usage: PYTHONPATH=<vllm-tt-plugin resident-mu src>:<tt-metal> python -m pytest --noconftest -c /dev/null \
    qwen36_resident/tests/test_vllm_rows_engine.py   (a venv with vLLM; no device is opened)
"""
import random
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

vllm_tt_plugin = pytest.importorskip("vllm_tt_plugin")
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

from qwen36_resident.tests.test_vllm_rows_fake import make_adapter, ref_next
from qwen36_resident.vllm_rows import BLOCK, CHUNK_UNIT, ResidentRowsAdapter, resident_capabilities

# a tiny model config of the plugin's own tests (vocab 128): the scheduler and runner need one, the fake ignores it
MODEL_CONFIG = Path(vllm_tt_plugin.__file__).resolve().parents[2] / "tests" / "model_configs" / "qwen2"
MAX_LEN = 4096


class ResidentModel(ResidentRowsAdapter):
    """the class vLLM resolves: the resident class's capabilities under the running plugin"""

    model_capabilities = resident_capabilities()

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return []


def vllm_config(max_num_seqs, async_scheduling, device_sampling):
    if not MODEL_CONFIG.is_dir():
        pytest.skip(f"no plugin test model config at {MODEL_CONFIG}")
    mc = ModelConfig(model=str(MODEL_CONFIG), dtype="float16", seed=42, skip_tokenizer_init=True)
    mc.max_model_len = MAX_LEN
    sc = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        max_model_len=MAX_LEN,
        enable_chunked_prefill=True,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    cc = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False)
    tt = {"sample_on_device_mode": "decode_only"} if device_sampling else {}
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch("vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups", return_value=None),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(ResidentModel, None)),
    ):
        return VllmConfig(
            scheduler_config=sc,
            model_config=mc,
            cache_config=cc,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": tt},
        )


class Engine:
    """vLLM's EngineCore loop over the real scheduler and runner (step / step_with_batch_queue)"""

    def __init__(self, max_num_seqs, async_scheduling, device_sampling, num_blocks):
        self.config = vllm_config(max_num_seqs, async_scheduling, device_sampling)
        self.vocab = self.config.model_config.get_vocab_size()
        kvc = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
                )
            ],
        )
        self.config.cache_config.num_gpu_blocks = num_blocks
        self.scheduler = TTScheduler(
            vllm_config=self.config,
            kv_cache_config=kvc,
            block_size=BLOCK,
            log_stats=False,
            structured_output_manager=StructuredOutputManager(self.config),
        )
        self.adapter, self.model = make_adapter(
            max_num_seqs, CHUNK_UNIT, device_sampling, self.vocab, MAX_LEN, cls=ResidentModel
        )
        self.pp = self.adapter.engine.pp._obj  # the raw FakePP (the adapter sees its contract view)
        self.runner = TTModelRunner(self.config, None, "all", False, 1)
        self.runner.model = self.adapter
        self.runner.initialize_kv_cache(kvc)
        self.async_scheduling = self.config.scheduler_config.async_scheduling
        self.queue = deque()
        self.outputs = {}  # req_id -> generated token ids
        self.finished = {}  # req_id -> finish status
        self.replays = {}  # req_id -> output indices produced by a replay prefill after a preemption
        self.steps = 0
        preempt = self.scheduler._preempt_request

        def record(request, *args, **kwargs):
            # the resumed prefill replays prompt + every output, those still in flight included (the TT scheduler
            # holds the request back until they arrive), and samples the next output from its logits
            n = request.num_output_tokens + request.num_output_placeholders
            self.replays.setdefault(request.request_id, []).append(n)
            return preempt(request, *args, **kwargs)

        self.scheduler._preempt_request = record

    def add(self, rid, prompt, max_tokens, seed=None):
        init_none_hash(sha256)
        if seed is None:
            params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
        else:
            params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.8, top_k=20, seed=seed)
        params.update_from_generation_config({}, eos_token_id=2)
        req = Request(
            request_id=rid,
            prompt_token_ids=list(prompt),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK, sha256),
        )
        self.scheduler.add_request(req)
        self.outputs[rid] = []

    def abort(self, rid):
        self.scheduler.finish_requests([rid], RequestStatus.FINISHED_ABORTED)
        self.finished[rid] = "aborted"

    def _update(self, so, out):
        if hasattr(out, "get_output"):
            out = out.get_output()
        for eco in self.scheduler.update_from_output(so, out).values():
            for o in eco.outputs:
                self.outputs[o.request_id] += list(o.new_token_ids)
                if o.finish_reason is not None and o.request_id not in self.finished:
                    self.finished[o.request_id] = str(o.finish_reason)

    def _execute(self, so):
        """execute_model, then sample_tokens when it deferred sampling (None): EMPTY output for an empty step"""
        out = self.runner.execute_model(so)
        return self.runner.sample_tokens(None) if out is None else out

    def step(self):
        """one EngineCore step; returns False when nothing is left"""
        self.steps += 1
        if not self.async_scheduling:
            if not self.scheduler.has_requests():
                return False
            so = self.scheduler.schedule()
            self._update(so, self._execute(so))
            return True
        # step_with_batch_queue, batch queue of 2
        if self.scheduler.has_requests():
            so = self.scheduler.schedule()
            self.queue.appendleft((so, self._execute(so)))
            if len(self.queue) < 2 and (so.total_num_scheduled_tokens > 0 or self.scheduler.has_requests()):
                return True
        elif not self.queue:
            return False
        so, out = self.queue.pop()
        self._update(so, out)
        return True


def reference(prompt, n, seed, vocab, replays=()):
    seq, out = list(prompt), []
    for i in range(n):
        if seed is None or i == 0 or i in replays:  # host-sampled from a prefill's (peaked) logits
            t = ref_next(seq, None, vocab)
        else:
            t = ref_next(seq, (seed, len(seq) - 1), vocab)
        out.append(t)
        seq.append(t)
    return out


LENGTHS = [1, 3, 40, 63, 64, 65, 200, 700, 1024, 1025, 1500, 2100, 3000]


def drive(
    max_num_seqs,
    async_scheduling,
    device_sampling,
    seed,
    num_blocks=None,
    n_requests=24,
    lengths=LENGTHS,
    max_tokens=(1, 60),
    arrival=0.3,
):
    rng = random.Random(seed)
    if num_blocks is None:
        num_blocks = max_num_seqs * MAX_LEN // BLOCK + 1
    eng = Engine(max_num_seqs, async_scheduling, device_sampling, num_blocks)
    reqs = {}
    pending = deque()
    for i in range(n_requests):
        n = rng.choice(lengths)
        prompt = [rng.randrange(3, eng.vocab) for _ in range(n)]
        sd = rng.randrange(1000) if device_sampling and rng.random() < 0.5 else None
        reqs[f"r{i}"] = dict(prompt=prompt, max_tokens=rng.randint(*max_tokens), seed=sd)
        pending.append(f"r{i}")
    aborts = set(rng.sample(sorted(reqs), 2))
    for _ in range(20000):
        if pending and rng.random() < arrival:
            rid = pending.popleft()
            r = reqs[rid]
            eng.add(rid, r["prompt"], r["max_tokens"], r["seed"])
        for rid in list(aborts):
            if rid not in eng.finished and len(eng.outputs.get(rid, [])) >= 3:
                eng.abort(rid)
                aborts.discard(rid)
        if not eng.step() and not pending:
            break
    else:
        raise AssertionError("the engine did not drain")
    while eng.queue:
        eng.step()
    done = 0
    for rid, r in reqs.items():
        assert rid in eng.finished, f"{rid} never finished"
        if eng.finished[rid] == "aborted":
            continue
        want = reference(r["prompt"], r["max_tokens"], r["seed"], eng.vocab, eng.replays.get(rid, ()))
        assert eng.outputs[rid] == want, f"{rid} (prompt {len(r['prompt'])}): {eng.outputs[rid]} != {want}"
        done += 1
    return eng, done


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("max_num_seqs", [1, 8, 16])
def test_host_sampling(max_num_seqs, async_scheduling, seed):
    eng, done = drive(max_num_seqs, async_scheduling, False, seed)
    assert done >= 20
    chunked = eng.config.scheduler_config.enable_chunked_prefill
    assert chunked == (max_num_seqs > 1), "the chunk core: off for one user, on for several"
    if max_num_seqs == 1:
        assert eng.pp.spans == 0, "one user: every prompt runs whole"
    else:
        assert eng.pp.interleaved > 0, "several users: decode launches run between a split prompt's spans"


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("max_num_seqs", [1, 8])
def test_device_sampling(max_num_seqs, async_scheduling, seed):
    eng, done = drive(max_num_seqs, async_scheduling, True, seed)
    assert done >= 20


@pytest.mark.parametrize("device_sampling", [False, True], ids=["host-sampled", "device-sampled"])
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_kv_pressure_preempts_and_replays(async_scheduling, device_sampling):
    """8 users with long outputs in a pool that cannot hold them all: vLLM preempts running requests and replays them
    (prompt + outputs through prefill, a fresh slot); every completed output stays exact"""
    eng, done = drive(
        8,
        async_scheduling,
        device_sampling,
        11,
        num_blocks=70,
        n_requests=16,
        lengths=[300, 500, 700],
        max_tokens=(200, 400),
        arrival=1.0,
    )
    assert done >= 14
    replayed = [rid for rid, at in eng.replays.items() if eng.finished[rid] != "aborted" and max(at) > 0]
    assert len(replayed) >= 3, f"the pool was meant to force preemptions mid-decode: {eng.replays}"
