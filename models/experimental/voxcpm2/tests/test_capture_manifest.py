# SPDX-License-Identifier: Apache-2.0
"""Pure host tests for artifact integrity; these do not verify CUDA inference."""
import json
import unittest
import tempfile
from pathlib import Path
import torch
from models.experimental.voxcpm2.validation.manifest import TensorRecorder, load_manifest, load_tensor, materialize_snapshot
from models.experimental.voxcpm2.validation.validate_capture import validate_capture, main
from models.experimental.voxcpm2.validation.capture_reference import capture_reference, build_parser


def make_capture(directory, *, oracle=True, offset=0):
    recorder = TensorRecorder(directory)
    value = torch.arange(4).float() + offset
    event = recorder.begin('component', (value,), {'flag': True})
    recorder.end(event, value)
    manifest = {'schema_version': 1, 'complete': True, 'oracle': 'official_voxcpm2_cuda' if oracle else None,
                'backend': 'cuda' if oracle else 'ttnn', 'cuda': {'device_name': 'fixture'},
                'source': {'revision': 'fixture'}, 'checkpoint': {'revision': 'fixture'},
                'generation': {'seed': 1}, 'tensors': recorder.tensors, 'events': recorder.events}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


class TestCaptureManifest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_snapshot_does_not_alias_mutated_buffers(self):
        tmp_path = self.directory
        recorder = TensorRecorder(tmp_path)
        value = torch.ones(4)
        event = recorder.begin('component', (value,), {})
        value.zero_()
        saved = torch.load(tmp_path / recorder.tensors['component/000000/args/0']['file'], weights_only=True)
        self.assertTrue(saved.equal(torch.ones(4)))
        recorder.end(event, value)


    def test_integrity_and_metadata(self):
        tmp_path = self.directory
        manifest = make_capture(tmp_path)
        load_manifest(tmp_path)
        key = next(iter(manifest['tensors']))
        self.assertTrue(load_tensor(tmp_path, manifest, key).equal(torch.arange(4).float()))
        (tmp_path / manifest['tensors'][key]['file']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_manifest(tmp_path)


    def test_requires_complete_cuda_oracle(self):
        tmp_path = self.directory
        manifest = make_capture(tmp_path, oracle=False)
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            load_manifest(tmp_path)
        manifest['complete'] = False
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_manifest(tmp_path, require_cuda=False)


    def test_validation_api_cli_and_error_threshold(self):
        tmp_path = self.directory
        ref, candidate = tmp_path / 'ref', tmp_path / 'candidate'
        make_capture(ref)
        make_capture(candidate, oracle=False, offset=1)
        self.assertTrue(validate_capture(ref, candidate)['passed'])
        self.assertTrue(not validate_capture(ref, candidate, max_abs=.1)['passed'])
        report = tmp_path / 'report.json'
        self.assertTrue(main(['--reference', str(ref), '--candidate', str(candidate), '--max-abs', '.1', '--report', str(report)]) == 1)
        self.assertTrue(json.loads(report.read_text())['passed'] is False)


    def test_missing_tensors_fail(self):
        tmp_path = self.directory
        ref, candidate = tmp_path / 'ref', tmp_path / 'candidate'
        make_capture(ref)
        manifest = make_capture(candidate, oracle=False)
        del manifest['tensors'][next(iter(manifest['tensors']))]
        (candidate / 'manifest.json').write_text(json.dumps(manifest))
        self.assertTrue(not validate_capture(ref, candidate)['passed'])

    def test_cpu_fallback_rejected(self):
        args = build_parser().parse_args(['--checkpoint', str(self.directory), '--checkpoint-revision', 'a' * 40,
                                         '--output', str(self.directory / 'out'), '--text', 'hello', '--device', 'cpu'])
        with self.assertRaisesRegex(RuntimeError, 'CUDA is required'):
            capture_reference(args)
        self.assertFalse(args.output.exists())

    def test_nested_values_and_metadata(self):
        manifest = make_capture(self.directory)
        event = manifest['events'][0]
        args = materialize_snapshot(self.directory, manifest, event['args'])
        self.assertIsInstance(args, tuple)
        self.assertTrue(args[0].equal(torch.arange(4).float()))
        key = next(iter(manifest['tensors']))
        manifest['tensors'][key]['shape'] = [5]
        with self.assertRaisesRegex(ValueError, 'metadata'):
            load_tensor(self.directory, manifest, key)

    def test_generation_provenance_mismatch(self):
        ref, candidate = self.directory / 'ref', self.directory / 'candidate'
        make_capture(ref)
        manifest = make_capture(candidate, oracle=False)
        manifest['generation']['seed'] = 2
        (candidate / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'generation differs'):
            validate_capture(ref, candidate)
