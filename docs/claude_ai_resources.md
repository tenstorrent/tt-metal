# Claude AI & Claude Code - Essential Resources

> Curated links from r/ClaudeAI, r/ClaudeCode community discussions and ecosystem.
> Note: Direct Reddit URLs could not be crawled (Anthropic's web agent is blocked by Reddit). Links below are community guides, blog posts, and official docs that aggregate and reference Reddit discussions.

---

## Reading Roadmap

### Tier 1 — Foundation (read first, ~2 hours)

These 5 resources give you 80% of the value. They are the official sources that everything else builds on.

| # | Link | Why it matters |
|---|------|---------------|
| 1 | [Claude Code Best Practices (Official)](https://code.claude.com/docs/en/best-practices) | The single most important doc. Everything else builds on this. |
| 2 | [CLAUDE.md Best Practices (UX Planet)](https://uxplanet.org/claude-md-best-practices-1ef4f861ce7c) | How to structure the file that controls Claude's behavior in every session. |
| 3 | [Create Custom Subagents (Official)](https://code.claude.com/docs/en/sub-agents) | The canonical reference for agent architecture. |
| 4 | [Orchestrate Agent Teams (Official)](https://code.claude.com/docs/en/agent-teams) | Official delegation and teams doc. |
| 5 | [Prompting Best Practices (Anthropic)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) | XML tags, context placement, explicit instructions — the core techniques. |

### Tier 2 — Practical Patterns (~2 hours)

Real-world configs and battle-tested advice from production users.

| # | Link | Why it matters |
|---|------|---------------|
| 6 | [Best Practices for Subagents (PubNub)](https://www.pubnub.com/blog/best-practices-for-claude-code-sub-agents/) | Production-tested: single responsibility, tool scoping, pipeline patterns. |
| 7 | [Subagents: Common Mistakes (ClaudeKit)](https://claudekit.cc/blog/vc-04-subagents-from-basic-to-deep-dive-i-misunderstood) | What NOT to do. Saves you from repeating others' mistakes. |
| 8 | [45 Claude Code Tips (GitHub)](https://github.com/ykdojo/claude-code-tips) | Skim for tips relevant to your workflow (not all 45 matter equally). |
| 9 | [Subagents & Task Delegation (DEV Community)](https://dev.to/letanure/claude-code-part-6-subagents-and-task-delegation-k6f) | Concrete delegation walkthrough. |
| 10 | [How I Use Every Claude Code Feature (sshh.io)](https://blog.sshh.io/p/how-i-use-every-claude-code-feature) | One person's full workflow — shows how all the pieces connect. |

### Tier 3 — Deep Dives (pick based on interest)

**If you care most about prompt writing:**

| Link | Why |
|------|-----|
| [We Tested 25 Techniques, 5 Worked (DreamHost)](https://www.dreamhost.com/blog/claude-prompt-engineering/) | Empirical data on what actually works. |
| [Skills vs Subagents (TDS)](https://towardsdatascience.com/claude-skills-and-subagents-escaping-the-prompt-engineering-hamster-wheel/) | When to use skills instead of CLAUDE.md. |

**If you care most about multi-agent orchestration:**

| Link | Why |
|------|-----|
| [Hidden Multi-Agent System (paddo.dev)](https://paddo.dev/blog/claude-code-hidden-swarm/) | TeammateTool internals — 13 operations discovered in the binary. |
| [Multi-Agent Orchestration (sjramblings)](https://sjramblings.io/multi-agent-orchestration-claude-code-when-ai-teams-beat-solo-acts/) | When teams beat solo — decision framework. |
| [Swarm Orchestration Skill (Gist)](https://gist.github.com/kieranklaassen/4f2aba89594a4aea4ad64d753984b2ea) | Copy-paste swarm config you can use immediately. |

**If you want ready-made agent collections:**

| Link | Why |
|------|-----|
| [100+ Subagents (VoltAgent)](https://github.com/VoltAgent/awesome-claude-code-subagents) | Browse and pick what fits your workflow. |
| [Multi-agent orchestration repo (wshobson)](https://github.com/wshobson/agents) | Plug-and-play orchestration. |

### Tier 4 — Skip Unless Needed

These are low-signal or marketing-heavy. Only visit if you exhaust the above:
- `awesome-claude-prompts` repo — mostly generic prompts, not Claude Code specific
- `godofprompt.ai` / `promptbuilder.cc` — SEO-heavy, thin content
- `claudecode.run` / `subagents.cc` — marketing sites with minimal unique content
- Reddit aggregation article — summarizes what's already in Tiers 1-3

### Recommended Schedule

```
Day 1:  Tier 1 (#1-5)  — official docs foundation
Day 2:  Tier 2 (#6-7)  — practical patterns + anti-patterns
Day 3:  Tier 2 (#8-10) — tips + full workflow example
Day 4+: Tier 3         — pick based on what you're building
```

> **Key insight from the community**: Most people over-invest in prompt engineering and under-invest in CLAUDE.md structure and agent scoping. Get your CLAUDE.md right and your subagent tool permissions tight — that matters more than clever prompts.

---

## 1. Optimal Prompt Writing & Fine-Tuning Approaches

### Official Anthropic Docs
- [Prompting Best Practices (Anthropic)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) - Official guide: XML tags, explicit instructions, context placement
- [Prompt Engineering Overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) - Full prompt engineering docs
- [System Prompts Guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/system-prompts) - How to write effective system prompts
- [Prompt Engineering Best Practices (Blog)](https://claude.com/blog/best-practices-for-prompt-engineering) - Anthropic blog post on prompting

### CLAUDE.md Best Practices
- [CLAUDE.md Best Practices - 10 Sections to Include (UX Planet)](https://uxplanet.org/claude-md-best-practices-1ef4f861ce7c) - Practical structure for CLAUDE.md files
- [Claude Code Best Practices (Official Docs)](https://code.claude.com/docs/en/best-practices) - Official best practices including CLAUDE.md usage
- [Claude Code Best Practices (Community Guide)](https://rosmur.github.io/claudecode-best-practices/) - Community-curated best practices
- [claude-code-best-practice (GitHub)](https://github.com/shanraisshan/claude-code-best-practice) - GitHub repo with CLAUDE.md examples and patterns

### Prompt Engineering Guides & Tips
- [45 Claude Code Tips (GitHub - ykdojo)](https://github.com/ykdojo/claude-code-tips) - From basics to advanced, includes cutting system prompt in half
- [32 Claude Code Tips: From Basics to Advanced (Substack)](https://agenticcoding.substack.com/p/32-claude-code-tips-from-basics-to) - Comprehensive tip collection from community
- [15 Best Claude Code Prompts That Earn Me 30 Hours a Week](https://buildtolaunch.substack.com/p/best-claude-code-prompts) - Real productivity prompts
- [Claude Prompt Engineering: We Tested 25 Popular Practices (These 5 Worked)](https://www.dreamhost.com/blog/claude-prompt-engineering/) - Empirical testing of prompt techniques
- [Claude Prompt Engineering Best Practices 2026 (Prompt Builder)](https://promptbuilder.cc/blog/claude-prompt-engineering-best-practices-2026) - Checklist and templates
- [7 Advanced Prompt Techniques for Claude](https://creatoreconomy.so/p/claude-7-advanced-ai-prompting-tips) - Advanced techniques
- [How I Learned to Prompt Claude Code Better - Four Modes (Medium)](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd) - Practical modes of interaction
- [I Made Claude Code Think Before It Codes (DEV Community)](https://dev.to/_vjk/i-made-claude-code-think-before-it-codes-heres-the-prompt-bf) - Think-first prompting approach
- [24 Claude Code Tips - Advent Calendar (DEV Community)](https://dev.to/oikon/24-claude-code-tips-claudecodeadventcalendar-52b5) - Daily tip collection

### Prompt Collections & Repos
- [awesome-claude-prompts (GitHub)](https://github.com/langgptai/awesome-claude-prompts) - Curated prompt collection
- [Claude Code System Prompts (GitHub - Piebald-AI)](https://github.com/Piebald-AI/claude-code-system-prompts) - All parts of Claude Code's system prompt, 18 tool descriptions, sub-agent prompts
- [Claude Prompt Engineering Guide (GitHub)](https://github.com/ThamJiaHe/claude-prompt-engineering-guide) - Comprehensive guide with MCP, Skills, and Superpowers integration

### Key Takeaways from Community
- Keep CLAUDE.md under 200 lines; for each line ask "would removing this cause mistakes?"
- Use XML tags to structure prompts - Claude was trained on structured prompts
- Give explicit permission to express uncertainty (reduces hallucinations)
- Use skills instead of CLAUDE.md for domain-specific workflows (loaded on-demand, saves tokens)
- Five techniques that deliver measurable results: XML tags, extended thinking, explicit instructions, few-shot examples, context placement

---

## 2. Best Agent Architecture Designs

### Official Docs
- [Create Custom Subagents (Official)](https://code.claude.com/docs/en/sub-agents) - Official subagent documentation
- [Orchestrate Agent Teams (Official)](https://code.claude.com/docs/en/agent-teams) - Official agent teams documentation
- [Building Agents with Claude Agent SDK (Anthropic)](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - Anthropic engineering blog on agent architecture
- [Modifying System Prompts (Agent SDK)](https://platform.claude.com/docs/en/agent-sdk/modifying-system-prompts) - System prompt customization for agents

### Architecture Guides & Deep Dives
- [Best Practices for Claude Code Subagents (PubNub)](https://www.pubnub.com/blog/best-practices-for-claude-code-sub-agents/) - Production-tested patterns: single responsibility, tool scoping, pipeline architecture
- [Claude Code Subagents: Practical Guide with Real Agent Configs (wmedia)](https://wmedia.es/en/writing/claude-code-subagents-guide-ai) - Real config examples
- [Claude Code Subagents: Common Mistakes & Best Practices (ClaudeKit)](https://claudekit.cc/blog/vc-04-subagents-from-basic-to-deep-dive-i-misunderstood) - Common pitfalls and how to avoid them
- [A Practical Guide to Subagents in Claude Code (eesel.ai)](https://www.eesel.ai/blog/subagents-in-claude-code) - Hands-on subagent guide
- [Claude Code's Hidden Multi-Agent System (paddo.dev)](https://paddo.dev/blog/claude-code-hidden-swarm/) - Discovering TeammateTool with 13 operations hidden in Claude Code binary
- [How I Use Every Claude Code Feature (sshh.io)](https://blog.sshh.io/p/how-i-use-every-claude-code-feature) - Comprehensive feature walkthrough including agents

### Multi-Agent Orchestration
- [Claude Code Multi-Agent Orchestration (sjramblings)](https://sjramblings.io/multi-agent-orchestration-claude-code-when-ai-teams-beat-solo-acts/) - When AI teams beat solo acts
- [Multi-agent orchestration for Claude Code 2026 (Shipyard)](https://shipyard.build/blog/claude-code-multi-agent/) - Production multi-agent patterns
- [Multi-Agent Orchestration: Running 10+ Claude Instances in Parallel (DEV Community)](https://dev.to/bredmond1019/multi-agent-orchestration-running-10-claude-instances-in-parallel-part-3-29da) - Scaling to 10+ parallel agents
- [From Tasks to Swarms: Agent Teams in Claude Code (alexop.dev)](https://alexop.dev/posts/from-tasks-to-swarms-agent-teams-in-claude-code/) - Evolution from tasks to swarms
- [Claude Code Agent Teams: Complete Guide 2026 (claudefast)](https://claudefa.st/blog/guide/agents/agent-teams) - Complete agent teams guide
- [Claude Code Swarm Orchestration Skill (GitHub Gist)](https://gist.github.com/kieranklaassen/4f2aba89594a4aea4ad64d753984b2ea) - Complete guide to multi-agent coordination with TeammateTool

### Agent Repos & Collections
- [awesome-claude-code-subagents (GitHub - VoltAgent)](https://github.com/VoltAgent/awesome-claude-code-subagents) - 100+ specialized subagents covering wide range of use cases
- [awesome-claude-agents (GitHub - rahulvrane)](https://github.com/rahulvrane/awesome-claude-agents) - Collection of awesome claude code subagents
- [agents - Multi-agent orchestration (GitHub - wshobson)](https://github.com/wshobson/agents) - Intelligent automation and multi-agent orchestration for Claude Code
- [ruflo - Agent orchestration platform (GitHub)](https://github.com/ruvnet/ruflo) - Multi-agent swarms, distributed swarm intelligence, RAG integration
- [Claude Code Agentrooms](https://claudecode.run/) - Multi-agent development workspace with @mentions routing

### Key Takeaways from Community
- Give each subagent ONE clear goal, input, output, and handoff rule
- Scope tools per agent (PM/Architect = read-heavy; Implementer = Edit/Write/Bash; Release = minimal)
- Pipeline architecture: analyst -> architect -> implementer -> tester -> security audit
- Orchestrator pattern: one coordinator for global planning, delegation, and state
- Start with 1-2 specialized agents, refine based on real projects, then scale

---

## 3. Delegation Policies

### Official Docs & Guides
- [Subagents and Task Delegation (DEV Community)](https://dev.to/letanure/claude-code-part-6-subagents-and-task-delegation-k6f) - Part 6 of a comprehensive Claude Code series
- [Subagents: How Claude Delegates Like Santa (DEV Community)](https://dev.to/rajeshroyal/subagents-how-claude-delegates-like-santa-gg3) - Delegation patterns explained
- [What Is Sub-Agent Delegation in Claude Code (ClaudeLog)](https://claudelog.com/faqs/what-is-sub-agent-delegation-in-claude-code/) - FAQ on delegation
- [Task/Agent Tools (ClaudeLog)](https://claudelog.com/mechanics/task-agent-tools/) - How task and agent tools interact
- [Claude Code Sub-Agent Delegation Setup (GitHub Gist)](https://gist.github.com/tomas-rampas/a79213bb4cf59722e45eab7aa45f155c) - Ready-to-use delegation setup

### Agent Teams Delegation
- [How to Set Up and Use Claude Code Agent Teams (Medium)](https://darasoba.medium.com/how-to-set-up-and-use-claude-code-agent-teams-and-actually-get-great-results-9a34f8648f6d) - Practical setup guide
- [Claude Agent Teams: Why AI Coding Is About to Feel Like Managing a Real Engineering Squad (Substack)](https://theexcitedengineer.substack.com/p/claude-agent-teams-why-ai-coding) - The engineering team metaphor
- [Agent Teams Workflow (GitHub - claude-code-ultimate-guide)](https://github.com/FlorianBruniaux/claude-code-ultimate-guide/blob/main/guide/workflows/agent-teams.md) - Workflow documentation
- [Claude Code Multiple Agent Systems: Complete 2026 Guide (eesel.ai)](https://www.eesel.ai/blog/claude-code-multiple-agent-systems-complete-2026-guide) - Full multi-agent guide
- [7 Claude Code Best Practices for 2026 (eesel.ai)](https://www.eesel.ai/blog/claude-code-best-practices) - Best practices from real projects

### Advanced Patterns
- [Designing Sub-Agents for Planning - Meet @architect (DEV Community)](https://dev.to/cristiansifuentes/conversational-development-with-claude-code-part-7-designing-sub-agents-for-planning-meet-1nlk) - The architect sub-agent pattern
- [Claude Skills and Subagents: Escaping the Prompt Engineering Hamster Wheel (TDS)](https://towardsdatascience.com/claude-skills-and-subagents-escaping-the-prompt-engineering-hamster-wheel/) - Skills vs subagents tradeoffs
- [How Sub-Agents Work in Claude Code (Medium)](https://medium.com/@kinjal01radadiya/how-sub-agents-work-in-claude-code-a-complete-guide-bafc66bbaf70) - Complete sub-agent internals guide
- [What Is Sub-Agent in Claude Code (ADevGuide)](https://adevguide.com/ai-engineering/llm-agents/what-is-sub-agent-in-claude-code/) - Developer guide
- [Claude Agent SDK Best Practices for AI Agent Development 2025 (skywork.ai)](https://skywork.ai/blog/claude-agent-sdk-best-practices-ai-agents-2025/) - SDK-level best practices
- [subagents.cc](https://subagents.cc/) - Dedicated Claude Code agent resource

### Key Delegation Principles from Community
- **Task-based > Role-based**: "API-endpoint-writer" beats "backend developer" for reliability
- **Explicit invocation > Auto-delegation**: In practice, explicitly calling subagents with clear instructions works better than relying on auto-detection
- **Context handoff matters**: The "handoff problem" - subagents start with a blank slate; provide detailed briefs to avoid "context amnesia"
- **Delegate mode (Shift+Tab)**: Toggle when the lead keeps coding instead of coordinating
- **Plan first, then delegate**: Two-step approach - plan in plan mode, then hand the plan to a team for parallel execution
- **Hooks for workflow gates**: Register SubagentStop hooks to enforce quality gates between delegation steps
- **Cost awareness**: More agents = more tokens = more cost; use teams only when coordination benefit justifies overhead
- **95% of tasks don't need multi-agent**: Reserve agent teams for genuinely large, parallelizable work

---

## Reddit Communities

- **r/ClaudeAI** - 280K+ members, general Claude discussion with prompt engineering focus
- **r/ClaudeCode** - 4,200+ weekly contributors, Claude Code specific workflows and tips
- **r/ChatGPTCoding** - Multi-tool comparison discussions
- **r/vibecoding** - 89K members, build logs and coding workflows
- **r/Anthropic** - Official Anthropic community

### Reddit Aggregation Article
- [Claude Code Reddit: What Developers Actually Use It For in 2026 (aitooldiscovery)](https://www.aitooldiscovery.com/guides/claude-code-reddit) - Aggregated analysis of 500+ Reddit comments across all Claude subreddits
