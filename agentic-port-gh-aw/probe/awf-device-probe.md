---
description: |
  THROWAWAY. Delete before this branch goes anywhere near main.

  Round 2. Round 1 established that the agent's own bash sees /dev/tenstorrent but gets EPERM on
  open() — AWF's container carries no --device rule. Every frontmatter escape from that
  (sandbox.agent: false, sandbox.agent.args) is rejected by strict mode, which is mandatory here
  because tt-metal is public.

  This round tests the remaining lead. gh-aw documents mcp-scripts as running "as an HTTP MCP
  server on the GitHub Actions runner, outside the agent container", reached via
  host.docker.internal. If that is true, an mcp-script tool inherits the runner host's device
  access, and the agent gets an in-loop measurement tool without ever holding the card itself.

  Three legs, one runner, so the comparison is exact:
    CONTROL   ordinary workflow step, on the host        -> known good
    TREATMENT agent's own bash, inside AWF               -> known EPERM
    MCPTOOL   mcp-script tool                            -> the question

on:
  push:
    branches: [ebanerjee/awf-device-probe]
  workflow_dispatch:

runs-on: [N150, in-service, cloud-virtual-machine]

timeout-minutes: 20

permissions:
  contents: read
  copilot-requests: write

engine: copilot
network: defaults

tools:
  bash: [":*"]

# The lead under test. If this executes on the runner host it can open the card; if it is
# containerised alongside the agent it will EPERM exactly like the agent's bash does.
# timeout is set well above the default 60s to check that a long-running measurement tool is
# even expressible — a real build-and-measure leg would need minutes, not seconds.
mcp-scripts:
  probe-device:
    description: >-
      Probe whether the caller can open a Tenstorrent card. Takes no meaningful input and returns
      JSON describing the execution context and the result of open() on each /dev/tenstorrent node.
      Call this exactly once.
    timeout: 300
    py: |
      import errno, glob, json, os, socket, stat

      result = {
          "leg": "MCPTOOL-mcp-script",
          "hostname": socket.gethostname(),
          "uid": os.getuid(),
          "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
          "cwd": os.getcwd(),
          "dockerenv_present": os.path.exists("/.dockerenv"),
          "awf_env_var_count": sum(1 for k in os.environ if k.startswith("AWF")),
          "nodes": [],
          "opens": [],
      }
      try:
          result["cgroup"] = open("/proc/self/cgroup").readline().strip()
      except Exception as e:
          result["cgroup"] = f"unreadable: {e}"

      nodes = sorted(glob.glob("/dev/tenstorrent/*"))
      result["nodes"] = nodes
      for n in nodes:
          try:
              st = os.stat(n)
              result["opens"].append({
                  "node": n,
                  "mode": oct(st.st_mode & 0o777),
                  "chardev": stat.S_ISCHR(st.st_mode),
                  "rdev": f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}",
              })
          except OSError as e:
              result["opens"].append({"node": n, "stat_error": str(e)})
          for flag, name in ((os.O_RDWR, "O_RDWR"), (os.O_RDONLY, "O_RDONLY")):
              try:
                  fd = os.open(n, flag)
                  os.close(fd)
                  result["opens"].append({"node": n, "mode_tried": name, "open": "OK"})
              except OSError as e:
                  result["opens"].append({
                      "node": n, "mode_tried": name, "open": "FAIL",
                      "errno": e.errno,
                      "errname": errno.errorcode.get(e.errno, "?"),
                      "strerror": e.strerror,
                  })

      ok = any(o.get("open") == "OK" for o in result["opens"])
      if not nodes:
          result["verdict"] = "NO_DEVICE_NODES_VISIBLE"
      elif ok:
          result["verdict"] = "OPENABLE — mcp-scripts run outside the agent container"
      else:
          codes = {o.get("errname") for o in result["opens"] if o.get("open") == "FAIL"}
          result["verdict"] = f"BLOCKED ({'/'.join(sorted(c for c in codes if c))})"
      print(json.dumps(result, indent=2))

steps:
  - name: Probe the card outside the sandbox (control)
    run: bash .github/scripts/awf-device-probe.sh CONTROL-outside-sandbox
---

# AWF device probe, round 2

Do exactly these two things, in order, then report.

**1. Re-confirm the sandbox result.** Run:

```
bash .github/scripts/awf-device-probe.sh TREATMENT-inside-sandbox
```

**2. Call the `probe-device` tool once.** It takes no inputs. It is an mcp-script tool, so it is
supposed to execute on the runner host rather than inside your container — that difference is the
entire point of this run.

Then write a report to your log with a three-row comparison. For the TREATMENT leg (your bash) and
the MCPTOOL leg (the tool), state for each:

- the hostname it reported, and whether it looked containerised (`/.dockerenv`, cgroup line, count of `AWF_*` variables)
- whether `/dev/tenstorrent` nodes were visible
- whether `open()` succeeded, and if not the exact errno and symbolic name

Finish with one sentence answering: **did the mcp-script tool open the card when your own bash could
not?** Say plainly which of the two happened. Both outcomes are useful results.

Do not create an issue, do not open a pull request, do not comment anywhere — print the report and
stop. Do not retry with `sudo`, do not try to install anything, and do not attempt to work around a
denied open. If the tool is not offered to you at all, say so explicitly rather than simulating its
output; that too is a result.
