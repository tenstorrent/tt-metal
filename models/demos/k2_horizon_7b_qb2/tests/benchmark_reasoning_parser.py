"""Native high-reasoning parser for the K2-Horizon benchmark server.

The HF template places the opening marker in the prompt. DeepSeek's parser
already handles that implicit opener and preserves an unfinished reasoning
response as reasoning with no final answer.
"""

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class K2HorizonBenchmarkReasoningParser(DeepSeekR1ReasoningParser):
    @property
    def start_token(self) -> str:
        return "<ifm|think>"

    @property
    def end_token(self) -> str:
        return "</ifm|think>"


ReasoningParserManager.register_module("k2_horizon_benchmark", module=K2HorizonBenchmarkReasoningParser)
