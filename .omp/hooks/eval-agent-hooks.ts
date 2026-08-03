// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * omp hook: eval-agent nudges, ported from `.claude/scripts/hooks/`.
 *
 * This is the omp-side companion to the Claude Code agent hooks that
 * `.claude/scripts/hooks/enable_agent_hooks.sh` merges into
 * `.claude/settings.local.json`. Those Claude hooks cannot fire under omp
 * because (a) they read Claude's stdin JSON (`agent_type`, `session_id`,
 * `hook_event_name`), (b) they respond on stdout via
 * `hookSpecificOutput.additionalContext`, and (c) the batch pipeline spawns
 * `claude -p` subprocesses whose hook bus is Claude's, not omp's. This module
 * reimplements the four *portable* hooks against omp's `tool_call` /
 * `tool_result` `HookAPI`, so an interactive omp session editing this repo
 * gets the same test / test-failure / skill / subagent guidance the batch
 * pipeline's claude agents get.
 *
 * Scoped per-session, NOT per-agent: omp's `tool_call`/`tool_result` events
 * carry no `agent_type`, so agent-role gating is impossible at the hook layer.
 * In the main omp session that's correct — the human is the role. If an
 * external harness wraps one omp session per agent, it can set `OMP_EVAL_AGENT`
 * and this module gates on it (unset = always act, matching a main session).
 *
 * The two Claude *Stop* hooks (`block-if-uncommitted`, `capture-friction`) are
 * intentionally NOT here: omp's `agent_end` is observe-only and cannot veto a
 * stop the way Claude's `Stop` hook can with `exit 2`. Their blocking
 * enforcement lives in the pipeline driver (`eval/pipeline.py` phase exit),
 * which fires for every runtime — claude, codex, and any omp-driven flow.
 *
 * Registration: placing this file at `.omp/hooks/eval-agent-hooks.ts` is all
 * that's needed — omp's `hookCapability` discovers `.omp/hooks/**​*.ts` and
 * loads it through the extension runner. It no-ops outside a tt-metal repo
 * (`ttnn/ttnn/operations` must exist under ctx.cwd).
 */

import type { HookAPI } from "@oh-my-pi/pi-coding-agent/extensibility/hooks";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve } from "node:path";

// The `ToolResultEvent`/`HookContext` shapes ride the package's exported type
// surface; the runtime values are plain objects. We describe only the fields we
// read and narrow at the one boundary, keeping the rest of the body `unknown`.
type ToolContent = { type?: string; text?: string };
interface ToolResultLike {
  toolName?: string;
  input?: Record<string, unknown>;
  content?: ToolContent[];
  isError?: boolean;
  error?: string;
}
interface ToolCallLike {
  toolName?: string;
  input?: Record<string, unknown>;
}
interface HookCtxLike {
  cwd?: string;
  ui?: { notify?: (msg: string, type?: string) => void };
  sessionManager?: { getSessionFile?: () => string };
}

/** `run_safe_pytest|tt-probe` as a command token, after stripping quoted
 * strings so we don't match a mere mention of the wrapper inside a string
 * literal. The vetted regex from kw-test-fail.sh. */
const WRAPPER_RE =
  /(^|[;&|]|\$\()\s*([A-Za-z_][A-Za-z_0-9]*=\S+\s+)*(bash\s+)?(\S+\/)?(run_safe_pytest|tt-probe)\.sh(\s|$)/;

const SCOPED_PATHS = [
  "ttnn/cpp/ttnn/operations",
  "ttnn/ttnn/operations",
  "tests/ttnn/unit_tests/operations",
  "eval/golden_tests",
];

/** Git repo root at or above `start` (the bash hooks' $PWD-upward `.git` walk). */
function findRepoRoot(start: string): string | null {
  let dir = resolve(start);
  for (let i = 0; i < 64; i++) {
    if (existsSync(join(dir, ".git"))) return dir;
    const parent = resolve(dir, "..");
    if (parent === dir) return null;
    dir = parent;
  }
  return null;
}

/** Uncommitted changes in operation scope (mirrors block_if_uncommitted.sh). */
function hasUncommittedOpChanges(repoRoot: string): boolean {
  const g = (args: string[]): string => {
    try {
      return execSync(
        `git -C ${JSON.stringify(repoRoot)} ${args.join(" ")} -- ` +
          SCOPED_PATHS.map((s) => JSON.stringify(s)).join(" "),
        { stdio: ["ignore", "pipe", "ignore"] },
      ).toString().trim();
    } catch {
      return "";
    }
  };
  return !!(
    g(["diff", "--cached", "--name-only"]) ||
    g(["diff", "--name-only"]) ||
    g(["ls-files", "--others", "--exclude-standard"])
  );
}

/** Bash hooks' hang signal: triage report present and written <5 min ago. */
function recentTriageExists(repoRoot: string): boolean {
  try {
    const st = statSync(join(repoRoot, "generated", "tt-triage", "triage.txt"));
    return st.size > 0 && Date.now() - st.mtimeMs < 5 * 60 * 1000;
  } catch {
    return false;
  }
}

/** Stable per-session id for one-dose state files. omp exposes no `session_id`
 * on hook events; the session file path is the closest durable identity, with
 * a cwd-hash fallback so repeated sessions in one repo don't collide. */
function sessionKey(ctx: HookCtxLike): string {
  let src = "";
  try {
    src = ctx.sessionManager?.getSessionFile?.() ?? "";
  } catch { /* no session manager (headless) */ }
  if (!src) src = String(ctx.cwd ?? process.cwd());
  return createHash("sha1").update(src).digest("hex").slice(0, 12);
}

const RUN_ROOT_RE = /\/localdev\/[^/\s"']+\/\d{4}_\d{2}_\d{2}\/[^/\s"']+/g;

function runRoot(path: string): string | null {
  return path.match(RUN_ROOT_RE)?.[0] ?? null;
}

function integrityPolicy(): string {
  const path = process.env.OMP_EVAL_INTEGRITY_FILE;
  if (!path) return "";
  try {
    return readFileSync(path, "utf8").replace(
      /\{baseline_sha\}/g,
      process.env.OMP_EVAL_BASELINE_SHA ?? "HEAD",
    );
  } catch {
    return "";
  }
}

function insideCheckout(path: string, checkout: string): boolean {
  let target = resolve(path);
  try { target = realpathSync(target); } catch { /* non-existent output path */ }
  let root = resolve(checkout);
  try { root = realpathSync(root); } catch { /* checkout is expected to exist */ }
  const rel = relative(root, target);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

function pathInputs(input: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === "string" && /(^|_)(path|file|directory|cwd)$/.test(key)) out.push(value);
    if (Array.isArray(value) && /(paths|files|directories)$/.test(key)) {
      out.push(...value.filter((v): v is string => typeof v === "string"));
    }
  }
  return out;
}

export default function evalAgentHooks(pi: HookAPI): void {
  // Agent-role gate. Unset OMP_EVAL_AGENT → act (a main omp session).
  const agentRole = (process.env.OMP_EVAL_AGENT ?? "").trim();
  const gate = (role: string): boolean => !agentRole || agentRole === role;

  // First device-harness invocation → one-dose skill reminder, once per instance.
  let firedOnce = false;

  // Integrity is fail-closed for previous-run access. The literal path check
  // happens before every tool executes. Delegated tasks receive the complete
  // policy in both task and context fields so subagents inherit the boundary.
  pi.on("tool_call", async (rawEvent: unknown, rawCtx: unknown) => {
    if (process.env.OMP_EVAL_INTEGRITY !== "1") return;
    const event = rawEvent as ToolCallLike;
    const ctx = rawCtx as HookCtxLike;
    const input = event.input ?? {};
    const checkout = process.env.OMP_EVAL_CHECKOUT ?? ctx.cwd ?? process.cwd();
    const allowed = runRoot(checkout);
    const mentioned = JSON.stringify(input).match(RUN_ROOT_RE) ?? [];
    const forbidden = mentioned.find((root) => root !== allowed);
    if (forbidden) {
      return {
        block: true,
        reason: `Integrity policy blocked access to previous eval run: ${forbidden}`,
      };
    }
    if (event.toolName === "bash") {
      const command = String(input.command ?? "");
      if (/(^|[\s;|&()])\.\.\//.test(command)) {
        return { block: true, reason: "Integrity policy blocks shell traversal outside the checkout." };
      }
      const localdevPaths = command.match(/\/localdev\/[^\s;|&()"']+/g) ?? [];
      const outside = localdevPaths.find((path) => !insideCheckout(path, checkout));
      if (outside) {
        return { block: true, reason: `Integrity policy blocks shell access outside the checkout: ${outside}` };
      }
    } else if (event.toolName !== "task") {
      const outside = pathInputs(input)
        .map((path) => isAbsolute(path) ? path : resolve(ctx.cwd ?? checkout, path))
        .find((path) => !insideCheckout(path, checkout));
      if (outside) {
        return { block: true, reason: `Integrity policy blocks file access outside the checkout: ${outside}` };
      }
    }
    if (event.toolName === "task") {
      const policy = integrityPolicy();
      if (!policy) {
        return { block: true, reason: "Integrity policy file is unavailable; refusing subagent delegation." };
      }
      const prefix = `${policy}\n\nThe parent task follows:\n`;
      return {
        input: {
          ...input,
          context: prefix + String(input.context ?? ""),
          task: prefix + String(input.task ?? input.description ?? ""),
        },
      };
    }
    return;
  });

  // Cast the runtime events once at the boundary; each field we read is typed
  // on the interfaces above, so the body below needs no further guards.
  pi.on("tool_result", async (rawEvent: unknown, rawCtx: unknown) => {
    const event = rawEvent as ToolResultLike;
    const ctx = rawCtx as HookCtxLike;

    const repoRoot = findRepoRoot(ctx.cwd ?? process.cwd());
    if (!repoRoot || !existsSync(join(repoRoot, "ttnn", "ttnn", "operations"))) return;

    if (event.toolName === "bash") {
      const cmd = typeof event.input?.command === "string" ? event.input.command : "";
      if (!WRAPPER_RE.test(cmd.replace(/"[^"]*"/g, "").replace(/'[^']*'/g, ""))) return;

      if (!firedOnce) {
        firedOnce = true;
        const text =
          "First device-harness invocation this session. The debug loop has specialised tools:\n" +
          "  • /debug-ttnn-op — triage report; CB sync; init/reconfig rules; --dev asserts; DPRINT probing.\n" +
          "  • Agent(ttnn-static-analyzer) — fresh-context structural review of all kernels; ~2 fix attempts.\n" +
          "  • Agent(ttnn-expert-debugger) — DPRINT-driven runtime investigation when still stuck.\n" +
          "This reminder fires once per session.";
        return { content: [...(event.content ?? []), { type: "text", text } satisfies ToolContent] };
      }

      if (!event.isError) {
        if (!gate("ttnn-implementer")) return;
        const text = hasUncommittedOpChanges(repoRoot)
          ? "Tests PASSED. You MUST: 1) Log a test_run breadcrumb (status=pass, test path + params). 2) Commit your in-scope changes now before continuing."
          : "Tests PASSED. Log a test_run breadcrumb (status=pass, test path + params). No in-scope changes to commit.";
        return { content: [...(event.content ?? []), { type: "text", text } satisfies ToolContent] };
      }

      if (!gate("ttnn-implementer")) return;
      const errText = (event.error ?? "") + (event.content ?? []).map((c) => c.text ?? "").join("");
      const hang = /status code 2|exit code 2/.test(errText) || recentTriageExists(repoRoot);
      const text = hang
        ? "HANG DETECTED. You MUST: 1) Log a hang_detected breadcrumb (test path + CB state from triage). 2) Read the triage callstacks/watcher log above — which RISC-V is stuck, on which CB. 3) Consult /debug-ttnn-op (hang-triage) for the matching signal. 4) Log a hypothesis breadcrumb before changing code."
        : "Test FAILED. You MUST: 1) Log a test_run breadcrumb (status=fail, error summary). 2) Classify: numerical_error / compile_error / assert. 3) Consult /debug-ttnn-op for that failure class. 4) Log a hypothesis breadcrumb before changing code.";
      return { content: [...(event.content ?? []), { type: "text", text } satisfies ToolContent] };
    }

    // log-implementer-tool-use: Agent / Skill invocations → tool_use breadcrumb.
    const tool = event.toolName === "task" ? "Agent" : event.toolName;
    if ((tool === "Agent" || tool === "Skill") && gate("ttnn-implementer")) {
      const inp = event.input ?? {};
      const detail =
        tool === "Agent"
          ? { event: "tool_use", tool: "Agent",
              subagent: (inp.agent ?? inp.subagent_type ?? "claude") as string,
              description: (inp.task ?? inp.description ?? "") as string }
          : { event: "tool_use", tool: "Skill",
              skill: (inp.skill ?? "") as string, args: (inp.args ?? "") as string };
      const text = `You just invoked the ${tool} tool. Log a tool_use breadcrumb with ` +
        `append_breadcrumb.sh using this event JSON: ${JSON.stringify(detail)}`;
      return { content: [...(event.content ?? []), { type: "text", text } satisfies ToolContent] };
    }
    return;
  });

  // Friction nudge — observe-only under omp (agent_end cannot block the way
  // Claude's Stop exit-2 can); the blocking enforcement is in eval/pipeline.py.
  // This mirrors capture-friction.sh's recall and surfaces a completion nudge.
  pi.on("agent_end", async (_ev: unknown, rawCtx: unknown) => {
    const ctx = rawCtx as HookCtxLike;
    const repoRoot = findRepoRoot(ctx.cwd ?? process.cwd());
    if (!repoRoot) return;

    if (agentRole && !["ttnn-implementer", "ttnn-expert-debugger", "incremental-verifier"].includes(agentRole)) return;

    const state = join(tmpdir(), `eval_friction_nudge_${agentRole || "main"}_${sessionKey(ctx)}`);
    if (existsSync(state)) return;

    // Has any breadcrumb file under operations recorded a friction event?
    let found = false;
    try {
      const opsDir = join(repoRoot, "ttnn", "ttnn", "operations");
      for (const op of readdirSync(opsDir)) {
        const logs = join(opsDir, op, "agent_logs");
        if (!existsSync(logs)) continue;
        for (const f of readdirSync(logs)) {
          if (f.endsWith("_breadcrumbs.jsonl") &&
              /"event"\s*:\s*"friction"/.test(readFileSync(join(logs, f), "utf8"))) {
            found = true;
            break;
          }
        }
        if (found) break;
      }
    } catch { /* best-effort */ }
    if (found) return;

    try {
      writeFileSync(state, "1");
    } catch { /* best-effort */ }
    try {
      ctx.ui?.notify?.(
        "Friction check: record any durable helper/doc/prompt gap as a friction breadcrumb " +
          "(append_breadcrumb.sh <op_path> <agent> '{\"event\":\"friction\",...}'); if none, log '{\"event\":\"friction\",\"what\":\"none\"}'.",
        "info",
      );
    } catch { /* no UI */ }
  });
}
