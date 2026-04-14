# Logging Reference

Optional execution logging for debugging and tracking LLK code generation workflows.

## Overview

Logging is optional and disabled by default. When enabled, it records agent execution steps to help debug issues.

## Enabling Logging

```bash
cd codegen-bh
./scripts/logging/set_logging.sh {operation} --enable
```

## Checking Logging Status

```bash
./scripts/logging/check_logging.sh {operation}
# Exit code 0 = enabled, 1 = disabled
```

## Log Location

Logs are written to: `artifacts/{operation}_log.md`

## Logging Commands

### Initialize Log

```bash
./scripts/logging/init_log.sh {operation} {agent_name}
```

### Append Log Entry

```bash
./scripts/logging/append_log.sh {operation} {type} "{message}"
```

Types:
- `action` - What the agent is doing
- `result` - Outcome of an action
- `error` - Error encountered
- `hypothesis` - Theory about an issue
- `recovery` - Fix being attempted
- `complete` - Final status

### Example Usage in Agent

```bash
# Check if logging enabled
./scripts/logging/check_logging.sh gelu
if [ $? -eq 0 ]; then
    ./scripts/logging/init_log.sh gelu llk-analyzer
    ./scripts/logging/append_log.sh gelu action "Reading BH reference"
    ./scripts/logging/append_log.sh gelu result "Found kernel type: SFPU"
    ./scripts/logging/append_log.sh gelu complete "SUCCESS - Analysis complete"
fi
```

## Disabling Logging

```bash
cd codegen-bh
./scripts/logging/set_logging.sh {operation} --disable
```

## Log Format

```markdown
# Execution Log: {operation}

## Agent: {agent_name}
Started: {timestamp}

### Actions
- [action] Reading BH reference
- [result] Found kernel type: SFPU
- [action] Analyzing function signatures
- [complete] SUCCESS - Analysis complete

Completed: {timestamp}
```
