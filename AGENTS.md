# Agent Instructions

## Push discipline

Do not push every individual commit to tt-metal. Pushing to the same PR more than once within a 5-minute window is treated as spam: it burns PR Gate runner capacity for no review benefit, and is grounds for the PR being closed and a cooldown before you may open another one.

Batch your changes locally and push once you have a complete, reviewable commit — not after every edit. This extends [CONTRIBUTING.md](CONTRIBUTING.md)'s existing guidance to avoid pushing before you're ready for review: once you are ready and iterating, don't fragment that into a push per edit either.
