# SPDX-License-Identifier: Apache-2.0
"""Host-only request evidence test: no server, model imports or devices."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('agentic_batch_probe.py')
spec = importlib.util.spec_from_file_location('agentic_batch_probe_snapshot_test', SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RequestSnapshotTest(unittest.TestCase):
    def check_snapshot(self, transport_error):
        item = {'id': 'conversation-00', 'messages': [
            {'role': 'system', 'content': 'Preserve this role.'},
            {'role': 'user', 'content': 'Alice has seven apples.'}]}
        expected = dict(model=probe.MODEL, messages=copy.deepcopy(item['messages']),
            max_tokens=81920, temperature=0.0, top_p=1.0, top_k=1, seed=1234,
            return_token_ids=True, chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh'))
        sent = []
        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 12000)
            sent.append(request.data)
            if transport_error:
                raise OSError('fixture transport failure')
            return io.BytesIO(json.dumps({'choices': [{'token_ids': [1,2], 'finish_reason':'stop'}],
                                         'usage': {'completion_tokens':2}}).encode())
        with patch.object(probe.urllib.request, 'urlopen', side_effect=fake_urlopen):
            row = probe.request('http://unused.invalid', item, 81920, 12000)
        self.assertEqual(row['status'], 'error' if transport_error else 'completed')
        self.assertEqual(sent[0], json.dumps(expected).encode())
        first_saved = json.dumps(row['request'], sort_keys=True)
        first_hash = row['request_sha256']
        # Exactly the main() transition, plus a nested edit that a shallow copy misses.
        item['messages'] += [{'role':'assistant','content':'native answer'},
                             {'role':'user','content':'follow-up'}]
        item['messages'][0]['content'] = 'Caller changed nested state.'
        self.assertEqual(json.dumps(row['request'], sort_keys=True), first_saved)
        self.assertEqual(probe.sha(row['request']), first_hash)
        self.assertEqual(row['request'], expected)
        self.assertEqual(first_hash, probe.sha(json.loads(sent[0])))
        restored = json.loads(json.dumps(row))
        self.assertEqual(restored['request'], expected)
        self.assertEqual(probe.sha(restored['request']), first_hash)

    def test_completed_response_snapshot(self):
        self.check_snapshot(False)

    def test_failed_response_snapshot(self):
        self.check_snapshot(True)


if __name__ == '__main__':
    unittest.main()
