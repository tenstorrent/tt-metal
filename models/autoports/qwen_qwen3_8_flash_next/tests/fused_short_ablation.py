# SPDX-License-Identifier: Apache-2.0
"""Real-HF short-tail controls for routing-sensitive fusion changes."""

import argparse
import inspect
import textwrap
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests import test_functional_decoder as contract
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["vector_hc", "sparse_moe"], required=True)
args = parser.parse_args()
if args.variant == "vector_hc":
    source = textwrap.dedent(inspect.getsource(fused.FusedHyperConnection.read))
    branch = "    if x.shape[2] > 4:\n        return self._read_prefill(x)\n"
    assert source.count(branch) == 1, "Source changed; inspect the short/vector boundary before running"
    source = source.replace(branch, "")
    namespace = dict(fused.__dict__)
    exec(source, namespace)
    fused.FusedHyperConnection = type("VectorHC", (fused.FusedHyperConnection,), {"read": namespace["read"]})
else:
    fused.FusedMoE = fused._PackedMoE
output = Path(__file__).resolve().parents[1] / "doc/fused_decoder" / ("short_" + args.variant)
output.mkdir(parents=True, exist_ok=True)
with patch.object(contract, "FunctionalDecoder", fused.FusedDecoder), patch.object(contract, "EVIDENCE", output):
    contract.test_functional_decoder_contract(0, 1)
