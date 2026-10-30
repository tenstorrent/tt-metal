# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The composite's runner gate: cachew must be a strict no-op off the CI VLAN.

GitHub-hosted runners cannot reach cachew. On them the action must not probe,
send a token, install the Git wrapper or change any Git configuration, so every
cachew step has to be skipped by the runner before any process starts.
"""

from pathlib import Path
import re
import unittest


ACTION = Path(__file__).resolve().parents[1] / "action.yml"
CONTEXT = re.compile(r"\b(runner|env|inputs|steps)\.([\w.-]+)")


def steps():
    text = ACTION.read_text()
    body = text.split("\n  steps:\n", 1)[1]
    result = []
    for block in re.split(r"\n    - ", "\n" + body)[1:]:
        lines = block.splitlines()
        step = {"name": lines[0].removeprefix("name: ").strip(), "raw": block}
        for index, line in enumerate(lines):
            match = re.match(r"\s*(id|if|uses):\s*(.*)$", line)
            if not match or match[1] in step:
                continue
            key, value = match[1], match[2].strip()
            if value in (">-", ">", "|"):
                continued = []
                for following in lines[index + 1 :]:
                    if not following.startswith("        "):
                        break
                    continued.append(following.strip())
                value = " ".join(continued)
            step[key] = value
        result.append(step)
    return result


def evaluate(expression, context):
    """Evaluate the subset of GitHub expression syntax used in action.yml."""

    def lookup(match):
        value = context.get(f"{match[1]}.{match[2]}", "")
        return repr(str(value).lower())

    python = CONTEXT.sub(lookup, expression)
    python = re.sub(r"'([^']*)'", lambda m: repr(m[1].lower()), python)
    python = python.replace("&&", " and ").replace("||", " or ")
    return bool(eval(python, {"__builtins__": {}}, {}))  # noqa: S307 - fixed, test-only grammar


def runner(environment, os="Linux", cachew="auto", kill_switch=None):
    context = {"runner.environment": environment, "runner.os": os, "inputs.cachew": cachew}
    if kill_switch is not None:
        context["env.CACHEW_CHECKOUT"] = kill_switch
    return context


class RunnerGateTests(unittest.TestCase):
    def setUp(self):
        self.steps = steps()
        self.session = next(step for step in self.steps if step.get("id") == "session")
        self.checkout = next(step for step in self.steps if step.get("id") == "checkout")

    def active_steps(self, context):
        """Steps the runner would start for a given runner context."""
        session_runs = evaluate(self.session["if"], context)
        started = []
        for step in self.steps:
            if step is self.checkout:
                started.append(step["id"])
                continue
            condition = step.get("if")
            self.assertIsNotNone(condition, f"cachew step {step['name']!r} must be gated")
            local = dict(context)
            # A skipped session step publishes no outputs.
            local["steps.session.outputs.ready"] = "true" if session_runs else ""
            if evaluate(condition, local):
                started.append(step["id"])
        return started

    def test_github_hosted_runners_only_run_the_official_checkout(self):
        for environment in ("github-hosted", ""):
            with self.subTest(environment=environment):
                self.assertEqual(self.active_steps(runner(environment)), ["checkout"])

    def test_self_hosted_linux_runs_cachew_by_default(self):
        self.assertEqual(self.active_steps(runner("self-hosted")), ["session", "prefetch", "checkout", "configure"])

    def test_opt_out_and_unsupported_platforms_are_no_ops(self):
        for context in (
            runner("self-hosted", cachew="false"),
            runner("self-hosted", kill_switch="false"),
            runner("self-hosted", os="Windows"),
            runner("self-hosted", os="macOS"),
            runner("github-hosted", cachew="true", kill_switch="false"),
        ):
            with self.subTest(context=context):
                self.assertEqual(self.active_steps(context), ["checkout"])

    def test_explicit_opt_in_on_other_linux_runners(self):
        self.assertEqual(
            self.active_steps(runner("github-hosted", cachew="true")), ["session", "prefetch", "checkout", "configure"]
        )

    def test_official_checkout_is_unconditional_and_unmodified(self):
        self.assertNotIn("if", self.checkout)
        self.assertNotIn("\n      env:", self.checkout["raw"])
        self.assertIn("actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd", self.checkout["uses"])

    def test_outputs_default_to_false_when_cachew_is_skipped(self):
        text = ACTION.read_text()
        self.assertIn("value: ${{ steps.configure.outputs.configured || 'false' }}", text)
        self.assertIn("value: ${{ steps.prefetch.outputs.prefetched || 'false' }}", text)

    def test_action_needs_no_pin_besides_the_consumers_pin(self):
        # Everything runs from the action's own directory; the only remote
        # action is the pinned official checkout.
        text = ACTION.read_text()
        self.assertNotIn("tenstorrent/tt-metal/", text)
        self.assertEqual(re.findall(r"^\s*uses:\s*(\S+)", text, re.M), [self.checkout["uses"].split()[0]])
        self.assertIn('run: bash "$GITHUB_ACTION_PATH/session/prepare.sh"', self.session["raw"])
        self.assertFalse((ACTION.parent / "session" / "action.yml").exists())

    def test_ssh_key_is_not_exposed_to_the_prefetch_process(self):
        prefetch = next(step for step in self.steps if step.get("id") == "prefetch")
        self.assertNotIn("inputs.ssh-key }}", prefetch["raw"])


if __name__ == "__main__":
    unittest.main()
