# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Publishing a prefill model's KV chunk table alongside its declared cache layout.

A model declares its caches as a ``CacheLayoutModel`` (the cache tensor layout spec), from which
``to_chunk_map`` derives the KV chunk address table. Until a model's ``enable_env`` is set, the classic
(walked) table is still published and the derived one is checked against it with
``first_table_difference``; the model is written next to the table (``<path>.cache_layout.pb``), for a
consumer that derives the table itself, only when the two agree. With ``enable_env=1`` the derived table is
published without walking.
"""

import os

from loguru import logger

import ttnn

disaggregation = ttnn.experimental.disaggregation

# The model is written next to the table it derives: ``<table path><LAYOUT_MODEL_SUFFIX>``.
LAYOUT_MODEL_SUFFIX = ".cache_layout.pb"


class CacheLayoutNotDerivable(Exception):
    """The caches' layout is one the CacheLayoutModel cannot yet declare."""


def row_major_mesh_id(fnids, rows, cols, where):
    """The mesh of the chips ``fnids[row][col]``, which must number them row-major over (rows, cols), as
    ``to_chunk_map`` numbers a cache's chips."""
    mesh_id = int(fnids[0][0].mesh_id)
    for r in range(rows):
        for c in range(cols):
            if (int(fnids[r][c].mesh_id), int(fnids[r][c].chip_id)) != (mesh_id, r * cols + c):
                raise CacheLayoutNotDerivable(f"{where} numbers its chips other than row-major over its mesh")
    return ttnn.MeshId(mesh_id)


def _export(export, value, path):
    """Writes ``value`` to ``path`` through a temporary file, so a reader never sees a partial file."""
    tmp = f"{path}.tmp"
    export(value, tmp)
    os.replace(tmp, path)


def _derived_table(model, hosts):
    table = disaggregation.to_chunk_map(model)
    for fid, host_name in hosts:
        table.set_fabric_node_host(fid, host_name=host_name)
    return table


def publish_kv_chunk_table(*, path, walk, layout_model, hosts, enable_env, model_name):
    """Publishes the KV chunk table to ``path`` and returns ``path``.

    ``walk()`` builds the classic table. ``layout_model()`` declares the caches as a ``CacheLayoutModel``, or
    raises ``CacheLayoutNotDerivable`` for a layout it cannot declare. ``hosts`` are (fabric node id, host
    name) pairs naming the host of every chip the table uses. ``model_name`` names the model in the log."""
    model, not_derivable = None, None
    try:
        model = layout_model()
    except CacheLayoutNotDerivable as error:
        not_derivable = str(error)
    model_path = path + LAYOUT_MODEL_SUFFIX

    if os.getenv(enable_env) == "1":
        if model is None:
            raise RuntimeError(
                f"{enable_env}=1, but the {model_name} cache layout model is not derivable: {not_derivable}"
            )
        table = _derived_table(model, hosts)
        _export(disaggregation.export_cache_layout_model_to_protobuf_file, model, model_path)
        _export(disaggregation.export_to_protobuf_file, table, path)
        logger.info(
            f"[migration] {model_name} KV chunk address table derived from its cache layout model ({model_path}) "
            f"and serialized to {path} (configs={table.num_configs()}, entries={table.total_entries()})"
        )
        return path

    walked = walk()
    if model is None:
        logger.info(
            f"[migration] {model_name} cache layout model not derived ({not_derivable}); the walked table stands alone"
        )
    else:
        difference = disaggregation.first_table_difference(walked, _derived_table(model, hosts))
        if difference is None:
            _export(disaggregation.export_cache_layout_model_to_protobuf_file, model, model_path)
            logger.info(
                f"[migration] the {model_name} cache layout model derives the walked KV chunk table; wrote it to "
                f"{model_path}"
            )
        else:
            logger.error(
                f"[migration] the KV chunk table derived from the {model_name} cache layout model differs from the "
                f"walked table at {difference}; publishing the walked table without the model"
            )
    _export(disaggregation.export_to_protobuf_file, walked, path)
    return path
