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


def value(expression, context):
    """Evaluate the subset of GitHub expression syntax used in action.yml."""

    def lookup(match):
        value = context.get(f"{match[1]}.{match[2]}", "")
        return repr(str(value).lower())

    python = CONTEXT.sub(lookup, expression.removeprefix("${{").removesuffix("}}"))
    python = re.sub(r"'([^']*)'", lambda m: repr(m[1].lower()), python)
    python = python.replace("&&", " and ").replace("||", " or ")
    return eval(python, {"__builtins__": {}}, {})  # noqa: S307 - fixed, test-only grammar


def evaluate(expression, context):
    return bool(value(expression, context))


def output(name):
    text = ACTION.read_text().split("\noutputs:\n", 1)[1].split("\nruns:\n", 1)[0]
    return re.search(rf"^  {name}:\n(?:    .*\n)*?    value: (.*)$", text, re.M)[1]


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

    def active_steps(self, context, cloned=False):
        """Steps the runner would start for a given runner context and cachew outcome."""
        local = dict(context)
        started = []
        for step in self.steps:
            condition = step.get("if")
            self.assertIsNotNone(condition, f"step {step['name']!r} must be gated")
            if evaluate(condition, local):
                started.append(step["id"])
                # A skipped step publishes no outputs.
                if step["id"] == "session":
                    local["steps.session.outputs.ready"] = "true"
                elif step["id"] == "cachew":
                    local["steps.cachew.outputs.checked-out"] = str(cloned).lower()
        return started

    def test_github_hosted_runners_only_run_the_official_checkout(self):
        for environment in ("github-hosted", ""):
            with self.subTest(environment=environment):
                self.assertEqual(self.active_steps(runner(environment)), ["checkout"])
                self.assertEqual(self.active_steps(runner(environment), cloned=True), ["checkout"])

    def test_self_hosted_linux_runs_cachew_by_default(self):
        self.assertEqual(self.active_steps(runner("self-hosted")), ["session", "cachew", "checkout", "configure"])

    def test_official_checkout_runs_only_when_cachew_did_not_check_out(self):
        self.assertEqual(self.active_steps(runner("self-hosted"), cloned=True), ["session", "cachew", "configure"])

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
            self.active_steps(runner("github-hosted", cachew="true")), ["session", "cachew", "checkout", "configure"]
        )

    def test_official_checkout_is_unmodified(self):
        self.assertEqual(self.checkout["if"], "steps.cachew.outputs.checked-out != 'true'")
        self.assertNotIn("\n      env:", self.checkout["raw"])
        self.assertIn("actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd", self.checkout["uses"])

    def test_outputs_default_to_false_when_cachew_is_skipped(self):
        self.assertEqual(output("cachew-configured"), "${{ steps.configure.outputs.configured || 'false' }}")
        self.assertEqual(output("cachew-checked-out"), "${{ steps.cachew.outputs.checked-out || 'false' }}")

    def test_ref_and_commit_come_from_whichever_step_checked_out(self):
        for name in ("ref", "commit"):
            cachew = {
                "steps.cachew.outputs.checked-out": "true",
                f"steps.cachew.outputs.{name}": "from-cachew",
            }
            fallback = {
                "steps.cachew.outputs.checked-out": "false",
                f"steps.cachew.outputs.{name}": "unused",
                f"steps.checkout.outputs.{name}": "from-checkout",
            }
            with self.subTest(name=name):
                self.assertEqual(value(output(name), cachew), "from-cachew")
                self.assertEqual(value(output(name), fallback), "from-checkout")
                self.assertEqual(value(output(name), {f"steps.checkout.outputs.{name}": "hosted"}), "hosted")
        # SHA input: checkout reports an empty ref, and so does cachew.
        self.assertEqual(value(output("ref"), {"steps.cachew.outputs.checked-out": "true"}), "")

    def test_action_needs_no_pin_besides_the_consumers_pin(self):
        # Everything runs from the action's own directory; the only remote
        # action is the pinned official checkout.
        text = ACTION.read_text()
        self.assertNotIn("tenstorrent/tt-metal/", text)
        self.assertEqual(re.findall(r"^\s*uses:\s*(\S+)", text, re.M), [self.checkout["uses"].split()[0]])
        self.assertIn('run: bash "$GITHUB_ACTION_PATH/session/prepare.sh"', self.session["raw"])
        self.assertFalse((ACTION.parent / "session" / "action.yml").exists())

    def test_ssh_key_is_not_exposed_to_the_cachew_checkout(self):
        cachew = next(step for step in self.steps if step.get("id") == "cachew")
        self.assertNotIn("inputs.ssh-key }}", cachew["raw"])


if __name__ == "__main__":
    unittest.main()
