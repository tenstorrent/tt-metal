# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.llama31_8b_p100_spec.tt import eagle_head
from models.demos.llama31_8b_p100_spec.tt import ops_spec_rows as spec_rows
from models.demos.llama31_8b_p100_spec.tt.readback import PipelinedDecode
from models.demos.llama31_8b_p100_spec.tt.spec_decode import Draft, draft_config, sampling_params
from models.ttt_compat.tt.common import Mode, PagedAttentionConfig
from models.ttt_compat.tt.generator import Generator

CONFIG = draft_config("eagle")
FEATURE_LAYERS = tuple(CONFIG["feature_layers"])
PROMPT_ROWS = CONFIG["prompt_rows"]
STEP_ROWS = CONFIG["step_rows"]
KV_BLOCKS = CONFIG["kv_blocks"]


def _zeros(mesh_device, rows, dim):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_dram_bf16(t):
    made = []
    if t.dtype != ttnn.bfloat16:
        t = ttnn.typecast(t, ttnn.bfloat16)
        made.append(t)
    if t.is_sharded() or t.memory_config().buffer_type != ttnn.BufferType.DRAM:
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
        made.append(t)
    return t, made


class TappedLayer:
    def __init__(self, layer, step_buffer, prompt_buffer):
        d = self.__dict__
        d["_layer"], d["_step"], d["_prompt"] = layer, step_buffer, prompt_buffer

    def __call__(self, x, *args, **kwargs):
        if kwargs.get("mode", Mode.DECODE) == Mode.DECODE:
            t, made = _to_dram_bf16(x)
            ttnn.copy(t, self._step)
            for m in made:
                ttnn.deallocate(m)
        else:
            shape = [int(v) for v in x.shape]
            if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] >= PROMPT_ROWS:
                made = []
                t = x
                if shape[2] > PROMPT_ROWS:
                    t = ttnn.slice(x, [0, 0, 0, 0], [1, 1, PROMPT_ROWS, shape[3]])
                    made.append(t)
                t, more = _to_dram_bf16(t)
                ttnn.copy(t, self._prompt)
                for m in made + more:
                    ttnn.deallocate(m)
        return self._layer(x, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._layer, name)

    def __setattr__(self, name, value):
        setattr(self._layer, name, value)


class FeatureTaps:
    def __init__(self, mesh_device, model, dim, layers=FEATURE_LAYERS):
        self.step = [_zeros(mesh_device, STEP_ROWS, dim) for _ in layers]
        self.prompt = [_zeros(mesh_device, PROMPT_ROWS, dim) for _ in layers]
        for i, index in enumerate(layers):
            model.layers[index] = TappedLayer(model.layers[index], self.step[i], self.prompt[i])


class FeatureMap:
    def __init__(self, mesh_device, args, weight):
        n, k = weight.shape
        grid = args.dram_shard_core_grid_for_k_and_n(k, n)
        self.input_config = ttnn.create_sharded_memory_config(
            (STEP_ROWS, k // grid.num_cores),
            grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.program_config = args.dram_matmul_config(m=STEP_ROWS, k=k, n=n)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.weight = ttnn.as_tensor(
            weight.float().T.contiguous().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=args.create_dram_sharded_mem_config(k, n),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self, features):
        x = ttnn.concat(features, dim=3)
        xs = ttnn.to_memory_config(x, self.input_config)
        ttnn.deallocate(x)
        y = ttnn.linear(
            xs,
            self.weight,
            program_config=self.program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(xs)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(y)
        return out


class EagleDraft(Draft):
    def __init__(
        self, mesh_device, model, model_args, head_dir, model_dir, cache_dir, optimizations, rows, max_seq_len=8192
    ):
        if rows not in CONFIG["head_rows"]:
            raise ValueError("draft head: rows must be 2, 4 or 8")
        self.mesh_device = mesh_device
        self.rows = rows
        self.dim = model_args.dim
        self.taps = FeatureTaps(mesh_device, model, self.dim)
        paged = PagedAttentionConfig(block_size=32, max_num_blocks=KV_BLOCKS)
        args, self.head, self.kv, self.order, self.inverse, weights, _, _ = eagle_head.create(
            mesh_device, head_dir, model_dir, cache_dir, optimizations, max_seq_len, paged, batch=rows
        )
        self.features = FeatureMap(mesh_device, args, weights["fc.weight"])
        del weights
        self.page_table = torch.arange(KV_BLOCKS, dtype=torch.int32).reshape(1, -1).repeat(rows, 1)
        self.row_state = spec_rows.RowState(mesh_device, rows, self.page_table, self.kv, None, per_row_sdpa=False)
        self.head.row_positions = tuple(self.row_state.update_idx)
        self.row_pos = [-1] * rows
        self.gen = Generator([self.head], [args], mesh_device, tokenizer=model_args.tokenizer)
        self.sampling = sampling_params()
        self.fresh = True
        self.traced = False
        self.steps = 0
        self.redone = 0
        self._reset()

    def _reset(self):
        self.base = None
        self.pending = None
        self.chain_row = 0
        self.host_rows = None
        self.ready = False

    def _host_int(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _write_rows(self, positions):
        for i in range(self.rows):
            p = int(positions[i])
            if self.row_pos[i] == p:
                continue
            values = [-1] * self.rows
            values[i] = p
            ttnn.copy_host_to_device_tensor(self._host_int(values), self.row_state.update_idx[i])
            self.row_pos[i] = p

    def _rows_moved(self):
        self.row_pos = [None if p is None or p >= 0 else -1 for p in self.row_pos]

    def _set_hidden(self, rows):
        t = torch.zeros(1, 1, STEP_ROWS, self.dim)
        t[0, 0, : rows.shape[0]] = rows.float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
            self.head.hidden,
        )

    def _inputs(self, tokens, positions):
        ids = torch.tensor([int(self.inverse[int(t)]) for t in tokens], dtype=torch.int64).reshape(self.rows, 1)
        return ids, torch.tensor(positions, dtype=torch.int64)

    def _kwargs(self, reload):
        kw = dict(
            page_table=self.page_table,
            kv_cache=[self.kv],
            enable_trace=True,
            sampling_params=self.sampling,
            reload_inputs=reload,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=self.fresh,
        )
        self.fresh = False
        return kw

    def _token(self, out, row):
        t = out[0] if isinstance(out, tuple) else out
        return int(self.order[int(t.reshape(-1)[row])])

    def _prompt_hidden(self, length):
        out = []
        for i in range(0, length, STEP_ROWS):
            parts = [ttnn.slice(t, [0, 0, i, 0], [1, 1, i + STEP_ROWS, self.dim]) for t in self.taps.prompt]
            y = self.features(parts)
            for t in parts:
                ttnn.deallocate(t)
            out.append(ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0, 0].float())
            ttnn.deallocate(y)
        return torch.cat(out, 0)[:length]

    def start(self, prompt_tokens, prompt_len, first_token):
        self._reset()
        if prompt_len < 2 or prompt_len > PROMPT_ROWS:
            return
        hidden = self._prompt_hidden(prompt_len)
        self.base = 0
        positions = list(range(prompt_len - 1))
        for i in range(0, len(positions), self.rows):
            chunk = positions[i : i + self.rows]
            tokens, pos = [0] * self.rows, [-1] * self.rows
            tokens[: len(chunk)] = [prompt_tokens[p + 1] for p in chunk]
            pos[: len(chunk)] = chunk
            self._set_hidden(hidden[chunk])
            self._write_rows(pos)
            self.gen.decode_forward(*self._inputs(tokens, pos), **self._kwargs(True))
            self._rows_moved()
            self.traced = True
        self.chain_row = 0
        self.host_rows = hidden[prompt_len - 1 : prompt_len].clone()
        self.ready = True

    def propose(self, token, position, k):
        if not self.ready:
            return []
        row = self.chain_row
        if row >= self.rows:
            raise ValueError("draft head: more accepted tokens than rows")
        p = position - 1 - self.base
        tokens, pos = [0] * self.rows, [-1] * self.rows
        tokens[row], pos[row] = token, p
        redo = 0
        if self.pending is not None:
            redo_tokens, redo_positions = self.pending
            redo = len(redo_tokens)
            tokens[:redo], pos[:redo] = redo_tokens, redo_positions
            self.pending = None
        if self.host_rows is not None:
            self._set_hidden(self.host_rows)
            self.host_rows = None
        self._write_rows(pos)
        steps = max(1, k)
        piped = self.traced
        pipe = PipelinedDecode(self.gen, is_tokens=True)
        idle = self._inputs([0] * self.rows, [-1] * self.rows)
        got = []
        for i in range(steps):
            if i == 1:
                off = list(self.row_pos)
                off[:redo] = [-1] * redo
                off[row] = self.row_pos[row]
                self._write_rows(off)
            args, reload = (self._inputs(tokens, pos), True) if i == 0 else (idle, False)
            if piped:
                out = pipe.step(*args, **self._kwargs(reload))
                if out is not None:
                    got.append(self._token(out, row))
            else:
                got.append(self._token(self.gen.decode_forward(*args, **self._kwargs(reload)), row))
                self.traced = True
        if piped:
            got.append(self._token(pipe.flush(), row))
        self._rows_moved()
        self.steps += steps
        self.redone += redo
        return got[:k] if k > 0 else []

    def commit(self, position, accepted, verify_tokens):
        hidden = self.features(self.taps.step)
        ttnn.copy(hidden, self.head.hidden)
        ttnn.deallocate(hidden)
        if self.base is None:
            self.base = position + accepted
        elif accepted:
            self.pending = (
                [int(verify_tokens[j]) for j in range(accepted)],
                [position + j - self.base for j in range(accepted)],
            )
        else:
            self.pending = None
        self.chain_row = accepted
        self.host_rows = None
        self.ready = True
