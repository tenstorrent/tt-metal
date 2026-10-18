# SPDX-License-Identifier: Apache-2.0
"""Exercise actual setup sharing with CPU metadata stand-ins; no TTNN import."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def share_function(name="_share_serving_constants", *, method=False):
    source = Path(__file__).resolve().parents[1] / "tt/model.py"
    tree = ast.parse(source.read_text())
    body = (
        next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenFlashNextModel")
        if method
        else tree.body
    )
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"json": json, "ttnn": SimpleNamespace(bfloat16="bf16", ROW_MAJOR_LAYOUT="rm", TILE_LAYOUT="tile")}
    namespace["ModelCache"] = ModelCache
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


class Tensor:
    def __init__(self, shape, *, padded=None, dtype="bf16", layout="tile", memory="dram"):
        self.shape = tuple(shape)
        self.padded_shape = self.shape if padded is None else tuple(padded)
        self.dtype, self.layout, self.memory = dtype, layout, memory

    def memory_config(self):
        return self.memory


class ModelCache:
    def __init__(self, layers):
        self.layers = layers


def gdn(mesh, **kwargs):
    return SimpleNamespace(
        mesh=mesh,
        kind="linear_attention",
        attention=SimpleNamespace(
            zero_recurrent=Tensor((32, 12, 128, 128), dtype="fp32", **kwargs),
            zero_conv=Tensor((32, 3, 2560), padded=(32, 32, 2560)),
            recurrent_state=Tensor((32, 12, 128, 128), dtype="fp32"),
            recurrent_bank=Tensor((64, 12, 128, 128), dtype="fp32"),
            conv_state=Tensor((32, 3, 2560), padded=(32, 32, 2560)),
            conv_bank=Tensor((64, 3, 2560), padded=(64, 32, 2560)),
        ),
    )


def qsa(mesh, context=262144):
    return SimpleNamespace(
        mesh=mesh,
        kind="indexed_attention",
        attention=SimpleNamespace(
            max_context=context,
            cos=Tensor((context, 64), layout="rm"),
            sin=Tensor((context, 64), layout="rm"),
            pooled_keys=object(),
            pooled_cache=object(),
            packed_cache=object(),
        ),
    )


def test_gdn_work_sharing_exports_shared_work_and_independent_banks():
    share, mesh, pool = share_function("_share_serving_gdn_work"), object(), {}
    first, second = gdn(mesh), gdn(mesh)
    banks = [(layer.attention.recurrent_bank, layer.attention.conv_bank) for layer in (first, second)]
    for layer in (first, second):
        layer.ple = None
        share(layer, pool)
    assert first.attention.recurrent_state is second.attention.recurrent_state
    assert first.attention.conv_state is second.attention.conv_state
    assert first.attention.recurrent_state is not first.attention.conv_state
    model = SimpleNamespace(layers=[first, second], serving_pool=True)
    cache = share_function("export_cache", method=True)(model)
    for index, row in enumerate(cache.layers):
        assert set(row) == {"recurrent_state", "recurrent_bank", "conv_state", "conv_bank"}
        assert row["recurrent_bank"] is banks[index][0]
        assert row["conv_bank"] is banks[index][1]
    for name in ("recurrent_state", "conv_state"):
        assert cache.layers[0][name] is cache.layers[1][name]
    assert cache.layers[0]["recurrent_bank"] is not cache.layers[1]["recurrent_bank"]
    assert cache.layers[0]["conv_bank"] is not cache.layers[1]["conv_bank"]


def test_external_cache_binding_preserves_supplied_references():
    share, export = share_function("_share_serving_gdn_work"), share_function("export_cache", method=True)
    mesh, pool = object(), {}
    layers = [gdn(mesh), gdn(mesh)]
    for layer in layers:
        layer.ple = None
        share(layer, pool)
    model = SimpleNamespace(layers=layers, serving_pool=True)
    model.cache = model._bound_cache = export(model)
    external_layers = [gdn(mesh), gdn(mesh)]
    for layer in external_layers:
        layer.ple = None
    external = export(SimpleNamespace(layers=external_layers, serving_pool=True))
    share_function("bind_cache", method=True)(model, external)
    assert model.cache is external and model._bound_cache is external
    for layer, supplied in zip(layers, external.layers):
        for name, tensor in supplied.items():
            assert getattr(layer.attention, name) is tensor
    assert layers[0].attention.recurrent_state is not layers[1].attention.recurrent_state


def test_gdn_work_metadata_and_mesh_guards():
    share, mesh = share_function("_share_serving_gdn_work"), object()
    for name in ("recurrent_state", "conv_state"):
        for changed in ("shape", "padded_shape", "dtype", "layout", "memory", "mesh"):
            pool, first, second = {}, gdn(mesh), gdn(mesh)
            tensor = getattr(second.attention, name)
            if changed == "mesh":
                second.mesh = object()
            else:
                setattr(tensor, changed, (1,) if changed in ("shape", "padded_shape") else "different")
            share(first, pool)
            share(second, pool)
            assert getattr(second.attention, name) is tensor, (name, changed)
            assert getattr(first.attention, name) is not tensor
    share(qsa(mesh), {})  # No GDN attributes are required for another layer kind.


def test_serving_requires_slots_before_binding_or_execution():
    forward = share_function("forward", method=True)

    class ReachedBinding(Exception):
        pass

    def bind_cache(cache):
        raise ReachedBinding

    for serving, slots in ((True, None), (True, object()), (False, None)):
        model = SimpleNamespace(serving_pool=serving, bind_cache=bind_cache)
        try:
            forward(model, None, positions=None, slots=slots, page_table=None, kv_cache=None)
        except ValueError as error:
            assert serving and slots is None
            assert "explicit request slots" in str(error)
        except ReachedBinding:
            assert not serving or slots is not None
        else:
            raise AssertionError("Expected slot rejection or arrival at cache binding")


def test_zero_sharing_preserves_all_mutable_owners():
    share, pool, mesh = share_function(), {}, object()
    first, second = gdn(mesh), gdn(mesh)
    mutable = {
        name: getattr(second.attention, name)
        for name in ("recurrent_state", "recurrent_bank", "conv_state", "conv_bank")
    }
    share(first, pool, {})
    share(second, pool, {})
    assert first.attention.zero_recurrent is second.attention.zero_recurrent
    assert first.attention.zero_conv is second.attention.zero_conv
    assert first.attention.zero_recurrent is not first.attention.zero_conv
    for name, value in mutable.items():
        assert getattr(second.attention, name) is value


def test_zero_metadata_and_mesh_guards():
    share, mesh = share_function(), object()
    for changed in ("shape", "padded_shape", "dtype", "layout", "memory", "mesh"):
        pool, first, second = {}, gdn(mesh), gdn(mesh)
        if changed == "mesh":
            second.mesh = object()
        else:
            value = (1,) if changed in ("shape", "padded_shape") else "different"
            setattr(second.attention.zero_recurrent, changed, value)
        original = second.attention.zero_recurrent
        share(first, pool, {})
        share(second, pool, {})
        assert second.attention.zero_recurrent is original, changed
        assert first.attention.zero_recurrent is not original


def test_rope_semantics_and_context_guards():
    share, pool, mesh = share_function(), {}, object()
    cfg = {"rope_type": "default", "rope_theta": 10000.0}
    first, second = qsa(mesh), qsa(mesh)
    mutable = second.attention.pooled_keys
    share(first, pool, cfg)
    share(second, pool, dict(reversed(list(cfg.items()))))
    assert first.attention.cos is second.attention.cos
    assert first.attention.sin is second.attention.sin
    assert first.attention.cos is not first.attention.sin
    assert second.attention.pooled_keys is mutable
    different_theta, different_context = qsa(mesh), qsa(mesh, 2048)
    before_theta, before_context = different_theta.attention.cos, different_context.attention.cos
    share(different_theta, pool, dict(cfg, rope_theta=50000.0))
    share(different_context, pool, cfg)
    assert different_theta.attention.cos is before_theta
    assert different_context.attention.cos is before_context
    assert before_theta is not first.attention.cos
    assert before_context is not first.attention.cos


def test_moe_scratch_identity_and_guards():
    share, mesh = share_function("_share_serving_moe_capacity"), object()

    def layer():
        return SimpleNamespace(
            mesh=mesh, mlp=SimpleNamespace(capacity=Tensor((1, 1, 4096, 2560), layout="rm"), weights=object())
        )

    pool, first, second = {}, layer(), layer()
    weights = second.mlp.weights
    share(first, pool)
    share(second, pool)
    assert first.mlp.capacity is second.mlp.capacity
    assert second.mlp.weights is weights
    for changed in ("shape", "padded_shape", "dtype", "layout", "memory", "mesh"):
        third = layer()
        if changed == "mesh":
            third.mesh = object()
        else:
            setattr(third.mlp.capacity, changed, (1,) if changed in ("shape", "padded_shape") else "different")
        original = third.mlp.capacity
        share(third, pool)
        assert third.mlp.capacity is original, changed
        assert original is not first.mlp.capacity


def test_paged_qsa_work_sharing_preserves_private_cache_and_external_binding():
    share, mesh, pool = share_function("_share_serving_qsa_work"), object(), {}

    def layer():
        result = qsa(mesh)
        result.ple = None
        result.attention.paged_pooled = True
        result.attention.blocks = 65536
        result.attention.pooled_keys = Tensor((1, 1, 65536, 128))
        for name in ("packed_cache", "raw_key_cache", "pooled_cache"):
            setattr(result.attention, name, Tensor((1, 1, 64, 128)))
        return result

    first, second = layer(), layer()
    private = {name: getattr(second.attention, name) for name in ("packed_cache", "raw_key_cache", "pooled_cache")}
    share(first, pool)
    share(second, pool)
    assert first.attention.pooled_keys is second.attention.pooled_keys
    for name, tensor in private.items():
        assert getattr(second.attention, name) is tensor
        assert tensor is not getattr(first.attention, name)
    model = SimpleNamespace(layers=[first, second], serving_pool=True)
    export = share_function("export_cache", method=True)
    model.cache = model._bound_cache = export(model)
    assert model.cache.layers[0]["pooled_keys"] is model.cache.layers[1]["pooled_keys"]
    supplied = ModelCache([vars(layer().attention) for _ in range(2)])
    supplied.layers = [{key: row[key] for key in model.cache.layers[0]} for row in supplied.layers]
    share_function("bind_cache", method=True)(model, supplied)
    assert model.cache is supplied
    assert first.attention.pooled_keys is supplied.layers[0]["pooled_keys"]
    assert second.attention.pooled_keys is supplied.layers[1]["pooled_keys"]
    assert first.attention.pooled_keys is not second.attention.pooled_keys
    for changed in ("shape", "padded_shape", "dtype", "layout", "memory", "mesh", "paged_pooled"):
        third = layer()
        if changed == "mesh":
            third.mesh = object()
        elif changed == "paged_pooled":
            third.attention.paged_pooled = False
        else:
            setattr(third.attention.pooled_keys, changed, (1,) if changed in ("shape", "padded_shape") else "different")
        original = third.attention.pooled_keys
        share(third, pool)
        assert third.attention.pooled_keys is original, changed
        assert original is not pool["paged_qsa_scoring_work"][0][1]
    share(gdn(mesh), pool)
