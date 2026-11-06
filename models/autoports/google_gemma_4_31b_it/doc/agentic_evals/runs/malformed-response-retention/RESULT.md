# Malformed-response retention

Confirmed upstream mini2.2.8 loses the response when native action parsing raises FormatError. The source-backed repair is to retain the response under the existing feedback message’s `extra.failed_response` and re-raise the same exception. This keeps feedback and subsequent requests unchanged and avoids changing Harbor usage accounting.

The current stock installer accepts only a published mini package version; there is no supported mini Git-SHA selector. A TTI-only config change cannot repair2.2.8. No runtime override, wrapper or source patch was applied. Consume a repaired upstream release through the existing version field, or separately review a generic Harbor immutable-source capability.

Diagnosis is confirmed; repair and host behavioral acceptance remain pending external dependency work. Device cost0h, box0.10h. The original missing responses remain unavailable, and long current decode alone establishes no loop. See AUTODEBUG.md and AUTOFIX.md for exact source chain and acceptance tests.
