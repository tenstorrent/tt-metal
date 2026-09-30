"""The one place the skill starts a process.

A process is started only for a program named in PROGRAMS, with its arguments as a list and no shell, so no argument
is ever parsed by a shell. The exceptions are `shell_run()` and `shell_popen()`, which run a command line the user typed
(a build or test command, a --then step) through /bin/sh, exactly as `shell=True` would.
"""

import subprocess

PROGRAMS = {
    "claude": "claude",
    "cp": "cp",
    "gh": "gh",
    "git": "git",
    "grep": "grep",
    "python3": "python3",
    "sh": "/bin/sh",
}


def run(program, args, **kw):
    return subprocess.run([PROGRAMS[program], *args], **kw)


def popen(program, args, **kw):
    return subprocess.Popen([PROGRAMS[program], *args], **kw)


def shell_run(cmd, **kw):
    """Run a user-typed command line through /bin/sh; never pass it repository or GitHub content."""
    return run("sh", ["-c", cmd], **kw)


def shell_popen(cmd, **kw):
    """Start a user-typed command line through /bin/sh; never pass it repository or GitHub content."""
    return popen("sh", ["-c", cmd], **kw)
