# SPDX-License-Identifier: Apache-2.0
import json

from models.autoports.google_gemma_4_31b_it.tools.agentic_collect_gpqa import sha
from models.autoports.google_gemma_4_31b_it.tools.agentic_gpqa_live_digest import save_native_statement


def test_actual_request_choice_order_is_preserved(tmp_path):
    actual = [{"role": "user", "content": "Which particle?\nA. neutron\nB. proton\nC. electron"}]
    separately_shuffled = "Which particle?\nA. proton\nB. electron\nC. neutron"
    boundary = {"id": "chatcmpl-actual", "request": json.dumps({"messages": actual, "seed": 42})}
    result = save_native_statement(tmp_path, "87", boundary)
    saved = json.loads(open(result["statement_file"]).read())
    assert saved["messages_or_prompt"] == actual
    assert saved["messages_or_prompt"][0]["content"] != separately_shuffled
    assert saved["native_response_id"] == boundary["id"]
    assert saved["native_request_sha256"] == sha(boundary["request"])
    assert (tmp_path / result["statement_file"]).stat().st_mode & 0o777 == 0o600
