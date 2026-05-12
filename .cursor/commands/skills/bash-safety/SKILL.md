---
name: bash-safety
description: >-
  Enforce safe bash scripting practices when writing, reviewing, or fixing
  shell scripts. Covers quoting, arrays, conditionals, arithmetic, redirections,
  strict mode, and static analysis. Use when editing .sh/.bash files, reviewing
  shell scripts, fixing shellcheck warnings, or writing new bash code.
---

# Bash Safety & Best Practices

When writing or reviewing bash scripts, apply the rules below. For the
complete catalog of rules with detailed examples, see [reference.md](reference.md).

Sources: [BashPitfalls](https://mywiki.wooledge.org/BashPitfalls),
[Shellharden](https://github.com/anordal/shellharden/blob/master/how_to_do_things_safely_in_bash.md).

## Rule 0: Run ShellCheck

Before any manual review, run static analysis:

```bash
shellcheck -o all script.sh
```

`-o all` enables every optional check (including those not on by default).
Fix all errors and warnings before proceeding. ShellCheck catches the
majority of the pitfalls listed here automatically.

## Rule Categories (quick reference)

| # | Category | Key Principle |
|---|----------|---------------|
| 1 | [Quoting](#1-quoting) | Always quote expansions |
| 2 | [Arrays](#2-arrays) | Use real arrays, not strings |
| 3 | [Conditionals](#3-conditionals--tests) | `[` is a command; prefer `[[ ]]` |
| 4 | [Loops](#4-loops--iteration) | Use globs / `while read`, not `for in $(…)` |
| 5 | [Arithmetic](#5-arithmetic) | Use `(( ))` for math |
| 6 | [Command Substitution](#6-command-substitution) | Prefer `"$(…)"` over backticks |
| 7 | [Redirections & Pipes](#7-redirections--pipes) | Order matters; pipes create subshells |
| 8 | [Filenames & Paths](#8-filenames--paths) | Prefix with `./`, use `--` |
| 9 | [Output](#9-output) | Use `printf`, not `echo` |
| 10 | [Script Structure](#10-script-structure) | Hashbang, strict mode, nullglob |
| 11 | [Dangerous Patterns](#11-dangerous-patterns) | Avoid injection, validate input |

---

## 1. Quoting

**The single most important rule.** An unquoted variable undergoes word
splitting and pathname expansion (globbing). Always quote `"$var"` and
`"$(cmd)"`.

```bash
# BAD
cp $file $target
echo $msg

# GOOD
cp -- "$file" "$target"
printf '%s\n' "$msg"
```

Exceptions (quoting optional but never harmful): `$?`, `$$`, `$!`, `$#`,
`${#array[@]}`, right side of assignments (`a=$b`), inside `[[ ]]`,
and inside `case … in`.

## 2. Arrays

Use real arrays when you need a list. Never use whitespace-delimited strings.

```bash
# BAD
files="a b c"
for f in $files; do …; done

# GOOD
files=(a b c)
for f in "${files[@]}"; do …; done
```

Always iterate positional parameters with `"$@"`, never `$*` or `$@`.

## 3. Conditionals & Tests

`[` is a command (alias for `test`). Spaces around every argument are mandatory.
`[[ ]]` is a bash keyword with safer parsing.

```bash
# BAD – missing spaces, wrong operator
[bar="$foo"]
[ bar == "$foo" ]
[ "$foo" = bar && "$bar" = foo ]

# GOOD
[ "$bar" = "$foo" ]
[[ $foo = "$bar" && $bar = "$baz" ]]
```

Unquoted RHS in `[[ ]]` is treated as a glob pattern. Quote it for literal
comparison: `[[ $foo = "$bar" ]]`.

## 4. Loops & Iteration

Never parse `ls` or unquoted command substitutions in `for`.

```bash
# BAD
for f in $(ls *.mp3); do …; done

# GOOD – use globs
for f in ./*.mp3; do
    [ -e "$f" ] || continue
    …
done

# GOOD – iterate command output via process substitution
while IFS= read -r line; do
    …
done < <(some_command)
```

## 5. Arithmetic

Use `(( ))` for integer math. Never use `[[ $x > 7 ]]` (string comparison).

```bash
# BAD
[[ $foo > 7 ]]
[ $foo > 7 ]        # creates a file named "7"

# GOOD
(( foo > 7 ))
[ "$foo" -gt 7 ]    # POSIX alternative
```

Validate user-supplied values before using them in arithmetic contexts to
prevent code injection.

## 6. Command Substitution

Prefer `"$(cmd)"` over backticks. Always quote the result.

```bash
# BAD
dir=`dirname $f`
cd $(dirname "$f")

# GOOD
dir="$(dirname -- "$f")"
cd -P -- "$(dirname -- "$f")"
```

`local var=$(cmd)` masks the exit status of `cmd`. Separate declaration
from assignment:

```bash
local var
var=$(cmd)
rc=$?
```

## 7. Redirections & Pipes

Redirections are evaluated left to right. `2>&1` must come **after** the
stdout redirect:

```bash
# BAD – stderr still goes to terminal
somecmd 2>&1 >>logfile

# GOOD
somecmd >>logfile 2>&1
```

Each command in a pipeline runs in a subshell; variable changes are lost
after the loop:

```bash
# BAD – count stays 0
grep foo bar | while read -r; do ((count++)); done

# GOOD – process substitution keeps loop in current shell
while read -r; do
    ((count++))
done < <(grep foo bar)
```

Never read from and write to the same file in one pipeline:

```bash
# BAD – truncates file
sed 's/foo/bar/' file > file

# GOOD
sed -i 's/foo/bar/' file      # GNU sed
sed 's/foo/bar/' file > tmp && mv tmp file   # portable
```

## 8. Filenames & Paths

Filenames can contain spaces, dashes, newlines, and glob characters.

- Prefix relative paths with `./` to prevent dash-as-option: `cp "./$f" dest/`
- Use `--` to end option parsing: `rm -- "$file"`
- Use `-print0` / `read -d ''` with `find`:

```bash
while IFS= read -r -d '' file; do
    …
done < <(find . -type f -name '*.mp3' -print0)
```

## 9. Output

`echo` cannot safely print arbitrary data (leading `-n`, `-e` interpreted
as options; no `--` support in GNU echo).

```bash
# BAD – breaks if $var starts with -n, -e, etc.
echo "$var"

# GOOD
printf '%s\n' "$var"
```

Never use the variable as the format string:

```bash
# BAD – format string injection
printf "$var"

# GOOD
printf '%s' "$var"
```

## 10. Script Structure

### Hashbang

```bash
#!/usr/bin/env bash
```

Don't put options (`-euo pipefail`) in the hashbang; they can be overridden.

### Strict mode

```bash
set -euo pipefail
```

Caveats:
- `set -u` treats empty arrays as unset in bash < 4.4. Use a feature check
  or omit `-u` for older bash.
- `set -e` is ignored inside functions called as conditions (`f && …`), and
  inside command substitutions. Always add explicit error checks.
- `pipefail` can cause false failures when earlier pipeline stages exit
  due to SIGPIPE (e.g. `cmd | head`).

### Safer globbing

```bash
shopt -s nullglob globstar
```

`nullglob` prevents unmatched globs from being passed as literal strings.
`globstar` enables `**` recursive globbing.

### Dependency assertion

```bash
require() { hash "$@" || exit 127; }
require curl jq
```

## 11. Dangerous Patterns

| Pattern | Risk | Fix |
|---------|------|-----|
| `eval "$var"` | Code injection | Avoid eval; use arrays |
| `$(( array[$x] ))` | Injection via index | Validate `$x` first |
| `find -exec sh -c 'echo {}'` | Code injection | Use `sh -c '…' _ {}` with `"$1"` |
| `export CDPATH=…` | Breaks `cd` in child scripts | Don't export CDPATH |
| `echo "Hello!"` (interactive) | History expansion | Use `printf` or `set +H` |

---

## Remediation Workflow

When fixing an existing script:

1. Run `shellcheck -o all script.sh` and fix all findings
2. Quote every unquoted variable and command substitution
3. Replace `echo "$var"` with `printf '%s\n' "$var"`
4. Replace `for x in $(cmd)` with `while read` loops
5. Replace string-based lists with arrays
6. Add `set -euo pipefail` (with appropriate caveats)
7. Add `shopt -s nullglob` if globs are used
8. Add `--` after commands that accept options before variable args
9. Prefix relative paths with `./` where needed
10. Re-run `shellcheck -o all` and verify clean

For the complete rule reference with 40+ detailed rules and examples,
see [reference.md](reference.md).
