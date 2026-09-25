# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership-contract regression; no TTNN binary or device is imported.

Run directly with Python to avoid the repository's device-aware pytest conftest.
The fake models the close/in-use contract in MeshDeviceImpl::close_impl, not
hardware scheduling. test_audio_submesh.py has the matching device regression.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def _release_method():
    path = Path(__file__).resolve().parents[2] / "pipelines" / "ltx" / "pipeline_ltx.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LTXPipeline")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "release_audio_submesh"
    )
    namespace = {"ttnn": SimpleNamespace(synchronize_device=lambda mesh: mesh.synchronize())}
    # Execute the production method, with only its TTNN dependency replaced.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["release_audio_submesh"]


class Mesh:
    """Synchronize drains work; quiesce also gives up shared-queue ownership."""

    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.in_use = True  # GlobalSemaphore initialization alone marks the CQ in use.
        self.closed = False
        self.quiesce_count = 0
        if parent is not None:
            parent.children.append(self)

    def synchronize(self):
        # Finish does not clear FDMeshCommandQueue::in_use_.
        pass

    def quiesce_devices(self):
        self.quiesce_count += 1
        for child in self.children:
            child.quiesce_devices()
        self.in_use = False

    def close(self):
        if self.in_use:
            parent = self.parent
            while parent is not None:
                if parent.in_use:
                    raise RuntimeError("cq is in use by parent mesh")
                parent = parent.parent
            if any(child.in_use for child in self.children):
                raise RuntimeError("cq is in use by child submesh")
        self.closed = True


def _pipeline(*, submesh=True):
    parent = Mesh()
    child = Mesh(parent) if submesh else None
    vae_ccl = object()
    return SimpleNamespace(
        mesh_device=parent,
        _owned_audio_submesh=child,
        audio_mesh_device=child or parent,
        audio_ccl_manager=object() if submesh else vae_ccl,
        vae_ccl_manager=vae_ccl,
        _audio_adapter=object(),
    )


class AudioSubmeshLifecycleTest(unittest.TestCase):
    def test_synchronize_only_reproduces_close_failure(self):
        pipeline = _pipeline()
        child = pipeline._owned_audio_submesh
        child.synchronize()
        with self.assertRaisesRegex(RuntimeError, "in use by parent"):
            child.close()
        with self.assertRaisesRegex(RuntimeError, "in use by child"):
            pipeline.mesh_device.close()

    def test_release_allows_fixture_child_then_parent_close(self):
        pipeline = _pipeline()
        child = pipeline._owned_audio_submesh
        _release_method()(pipeline)
        self.assertIsNone(pipeline._audio_adapter)
        self.assertIs(pipeline.audio_mesh_device, pipeline.mesh_device)
        self.assertIs(pipeline.audio_ccl_manager, pipeline.vae_ccl_manager)
        self.assertIs(pipeline._owned_audio_submesh, child)
        self.assertFalse(child.closed)
        child.close()
        pipeline.mesh_device.close()
        self.assertTrue(child.closed and pipeline.mesh_device.closed)

    def test_parent_can_resume_before_fixture_closes_child(self):
        pipeline = _pipeline()
        _release_method()(pipeline)
        pipeline.mesh_device.in_use = True
        pipeline._owned_audio_submesh.close()
        pipeline.mesh_device.close()

    def test_repeated_release_is_safe(self):
        pipeline = _pipeline()
        release = _release_method()
        release(pipeline)
        release(pipeline)
        pipeline._owned_audio_submesh.close()
        pipeline.mesh_device.close()

    def test_full_mesh_is_unchanged(self):
        pipeline = _pipeline(submesh=False)
        before = dict(vars(pipeline))
        _release_method()(pipeline)
        self.assertEqual(vars(pipeline), before)
        self.assertEqual(pipeline.mesh_device.quiesce_count, 0)

    def test_failed_quiesce_keeps_resources_reachable(self):
        pipeline = _pipeline()
        before = dict(vars(pipeline))

        def fail():
            raise RuntimeError("quiesce failed")

        pipeline.mesh_device.quiesce_devices = fail
        with self.assertRaisesRegex(RuntimeError, "quiesce failed"):
            _release_method()(pipeline)
        self.assertEqual(vars(pipeline), before)


if __name__ == "__main__":
    unittest.main()
