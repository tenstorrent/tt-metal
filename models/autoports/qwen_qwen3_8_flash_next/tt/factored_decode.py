# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded fixed-slot decode traces, preserving native decoder arithmetic.

Attention/cache addresses remain fixed. Only MoE activations are packed into
power-of-two row buckets, with a repeated valid row and masked output scatter.
N=1 keeps the original FP32 MoE path; all other buckets keep native BF16.
Evidence: factored_decode_buckets_{tracked,clean}_v1.json. Standalone B1 uses
its original whole-model trace and never constructs this owner.
"""

import time

from .indexed_rows import bucket_for_count, indexed_copy, metadata_words, validate_slots


class FactoredDecode:
    def __init__(self, generator, *, tt=None):
        if tt is None:
            import ttnn as tt
        import torch

        self.gen, self.tt = generator, tt
        self.slot_count = generator.max_batch_size
        if not 2 <= self.slot_count <= 32:
            raise ValueError("Factored decode owns2..32 fixed slots")
        self.buckets = tuple(1 << exponent for exponent in range((self.slot_count - 1).bit_length() + 1))
        self.traces = generator.traces
        self.active_slots = tuple(range(self.slot_count))
        self.metadata_uploads = self.segment_replays = 0
        self._plan_slots, self._plan = None, ()
        self._warmed = False
        width = generator.model.config.hidden_size

        def allocate(rows, features):
            return tt.zeros(
                (1, 1, rows, features),
                device=generator.mesh,
                dtype=tt.bfloat16,
                layout=tt.TILE_LAYOUT,
                memory_config=tt.DRAM_MEMORY_CONFIG,
            )

        def slots(features):
            return [allocate(1, features) for _ in range(self.slot_count)]

        self.residual, self.mixed, self.blocks, self.injection = slots(width), slots(width), slots(width), slots(1)
        self.packed = {bucket: allocate(bucket, width) for bucket in self.buckets}
        self.slot_indices = tt.from_torch(
            torch.tensor(metadata_words(self.active_slots, max_slots=self.slot_count), dtype=torch.int32),
            device=generator.mesh,
            dtype=tt.uint32,
            layout=tt.ROW_MAJOR_LAYOUT,
            memory_config=tt.DRAM_MEMORY_CONFIG,
        )

    def set_slots(self, slots):
        slots = tuple(slots)
        if slots == self.active_slots:
            return slots
        import torch

        slots = validate_slots(tuple(sorted(slots)))
        if slots[-1] >= self.slot_count:
            raise ValueError("Decode slot exceeds configured fixed-slot capacity")
        if slots != self.active_slots:
            self.gen._refresh(
                self.slot_indices, torch.tensor(metadata_words(slots, max_slots=self.slot_count), dtype=torch.int32)
            )
            self.active_slots = slots
            self.metadata_uploads += 1
        return slots

    def clear(self):
        self.tt.full_like(self.gen.sample_logits, -100.0, optional_tensor=self.gen.sample_logits)

    def attention(self, index, slot):
        tt, gen = self.tt, self.gen
        layer = gen.model.layers[index]
        words = None
        if index == 0 or layer.has_ple:
            token = tt.reshape(gen.decode_tokens[:, :, :, slot : slot + 1], (1, 1))
            if layer.has_ple:
                words = tt.concat([tt.reshape(token, (1, 1, 1)), gen.word_suffix], dim=2)
        if index == 0:
            x = gen.model.embed(token)
        else:
            previous = gen.model.layers[index - 1]
            x = previous.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        residual, mixed, injection = layer._attention_chunk(x, gen.caches[slot].layers[index], words)
        tt.copy(residual, self.residual[slot])
        tt.copy(mixed, self.mixed[slot])
        tt.copy(injection, self.injection[slot])

    def moe(self, index, bucket):
        packed = self.packed[bucket]
        indexed_copy(self.slot_indices, packed, self.mixed)
        blocks = self.gen.model.layers[index].mlp(packed)
        indexed_copy(self.slot_indices, blocks, self.blocks, scatter=True)

    def terminal(self, slot):
        from .chunk_output import write_token_chunk

        layer = self.gen.model.layers[-1]
        residual = layer.mlp_mix.write(self.residual[slot], self.blocks[slot], self.injection[slot])
        write_token_chunk(self.gen.model.logits(residual), self.gen.sample_logits, slot)

    def warmup(self):
        if self._warmed:
            return
        if self.traces:
            raise RuntimeError("Factored warmup must precede every model capture")

        def warm(key, body):
            started = time.perf_counter()
            body()
            self.gen._event("warmup", signature=("decode", key), host_submission_seconds=time.perf_counter() - started)

        for bucket in self.buckets:
            slots = tuple(range(min(bucket, self.slot_count)))
            for cache in self.gen.caches:
                self.gen._reset_cache(cache)
            self.set_slots(slots)
            warm(("clear",), self.clear)
            for index in range(len(self.gen.model.layers)):
                for slot in slots:
                    warm(("attention", index, slot), lambda: self.attention(index, slot))
                warm(("moe", index, bucket), lambda: self.moe(index, bucket))
            for slot in slots:
                warm(("terminal", slot), lambda: self.terminal(slot))
        self._warmed = True

    def capture(self):
        if not self._warmed:
            raise RuntimeError("Every factored shape must be warmed before capture")

        def record(key, body):
            if key not in self.traces:
                self.traces[key] = self.gen._record_trace(body, signature=("decode", key))
                self.gen.counters["captures"] += 1

        record(("clear",), self.clear)
        for index in range(len(self.gen.model.layers)):
            for slot in range(self.slot_count):
                record(("attention", index, slot), lambda index=index, slot=slot: self.attention(index, slot))
            for bucket in self.buckets:
                self.set_slots(tuple(range(min(bucket, self.slot_count))))
                record(("moe", index, bucket), lambda index=index, bucket=bucket: self.moe(index, bucket))
        for slot in range(self.slot_count):
            record(("terminal", slot), lambda slot=slot: self.terminal(slot))
        self.invalidate_plan()

    def replay(self, slots):
        # Close may release captures while retaining the already-warmed owner.
        if not self.traces:
            self.capture()
        slots = self.set_slots(slots)
        if slots != self._plan_slots:
            plan = [self.traces["clear",]]
            bucket = bucket_for_count(len(slots))
            for index in range(len(self.gen.model.layers)):
                plan.extend(self.traces["attention", index, slot] for slot in slots)
                plan.append(self.traces["moe", index, bucket])
            plan.extend(self.traces["terminal", slot] for slot in slots)
            self._plan_slots, self._plan = slots, tuple(plan)
        for trace in self._plan:
            self.tt.execute_trace(self.gen.mesh, trace, cq_id=0, blocking=False)
        self.segment_replays += len(self._plan)

    def invalidate_plan(self):
        self._plan_slots, self._plan = None, ()
