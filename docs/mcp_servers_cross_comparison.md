# MCP Server Cross-Comparison: DeepWiki vs Glean vs Atlassian

This document provides a three-way comparison of the MCP servers available in this project — **DeepWiki**, **Glean**, and **Atlassian** — covering their design philosophy, data access patterns, tool surface area, context efficiency, write capabilities, and decision guidance.

For individual deep-dives, see:
- [mcp_servers_explained.md](mcp_servers_explained.md) (DeepWiki)
- [glean_mcp_server_explained.md](glean_mcp_server_explained.md) (Glean)
- [atlassian_mcp_server_explained.md](atlassian_mcp_server_explained.md) (Atlassian)

---

## 1. High-Level Overview

| Aspect | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| **Purpose** | AI-powered documentation for GitHub repos | Enterprise knowledge search across all company tools | Direct API access to Confluence and Jira |
| **Server type** | RAG synthesis engine | Enterprise search + AI synthesis engine | Thin REST API proxy |
| **Data source** | GitHub repositories | SharePoint, Slack, Confluence, Jira, Google Drive, email, etc. | Confluence pages + Jira issues (Atlassian cloud only) |
| **Authentication** | None | Enterprise SSO | OAuth / API token |
| **Permission model** | Public repos (or configured private) | User's enterprise access permissions | User's Atlassian access permissions |
| **Tool count** | 3 | 3 | 30+ |
| **Write capability** | None (read-only) | None (read-only) | Full CRUD (create, read, update pages/issues/comments) |

### Design Philosophy

The three servers represent fundamentally different approaches to connecting an LLM to external knowledge:

- **DeepWiki** is a **synthesis-first** server. It ingests a GitHub repo, builds a knowledge index, and returns AI-generated answers. The LLM asks a question and gets a clean answer — no raw data wrangling needed.

- **Glean** is a **search-first** server. It indexes all enterprise knowledge sources and provides keyword search, document retrieval, and an AI chat interface. It straddles the line between raw retrieval and synthesis, but its responses are wrapped in heavy enterprise metadata.

- **Atlassian** is a **data-first** server. It exposes raw Atlassian REST API operations. There is no intelligence on the server side — no search ranking, no summarization, no synthesis. The LLM gets exactly what the API returns, verbatim.

---

## 2. Tool Surface Comparison

### Tool Count and Complexity

| Server | Tool Count | Tool Granularity | Parameter Complexity |
|---|---|---|---|
| DeepWiki | 3 | Coarse (one tool per task) | Simple (1-2 params each) |
| Glean | 3 | Coarse (one tool per task) | Moderate (1-14 params for `search`) |
| Atlassian | 30+ | Fine-grained (one tool per API endpoint) | High (2-8 params, `cloudId` required on all) |

DeepWiki and Glean each expose 3 tools that form a natural workflow. Atlassian exposes 30+ tools that mirror the Atlassian REST API, requiring Claude to understand Confluence vs. Jira data models, CQL vs. JQL query syntax, and multi-step navigation patterns.

### Functional Mapping

| Function | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| **Search / discover** | `read_wiki_structure` | `search` | `search` (Rovo), `searchConfluenceUsingCql`, `searchJiraIssuesUsingJql` |
| **Read full content** | `read_wiki_contents` | `read_document` | `getConfluencePage`, `getJiraIssue` |
| **Ask a question** | `ask_question` | `chat` | _(none — Claude must synthesize manually)_ |
| **Browse hierarchy** | _(flat topic list)_ | _(flat search results)_ | `getConfluenceSpaces` → `getPagesInConfluenceSpace` → `getConfluencePageDescendants` |
| **Read comments** | _(not applicable)_ | _(included if indexed)_ | `getConfluencePageFooterComments`, `getConfluencePageInlineComments`, `getConfluenceCommentChildren` |
| **Create content** | _(not available)_ | _(not available)_ | `createConfluencePage`, `createJiraIssue`, `createConfluenceFooterComment`, `createConfluenceInlineComment` |
| **Update content** | _(not available)_ | _(not available)_ | `updateConfluencePage`, `editJiraIssue`, `transitionJiraIssue` |
| **Manage workflow** | _(not available)_ | _(not available)_ | `getTransitionsForJiraIssue`, `transitionJiraIssue`, `addWorklogToJiraIssue` |
| **People lookup** | _(not available)_ | _(included in search metadata)_ | `lookupJiraAccountId`, `atlassianUserInfo` |

Key observations:
- DeepWiki and Glean both have a "ask a question and get an answer" tool. Atlassian does not — Claude must orchestrate multiple raw API calls and synthesize the answer itself.
- Only Atlassian supports **write operations** (creating/updating pages, issues, comments, transitions).
- Atlassian is the only server that supports **hierarchical navigation** (spaces → pages → children → comments).

---

## 3. Data Access Patterns

### How Many Tool Calls to Answer a Question?

Consider the question: _"What is the Tensix ISA?"_

| Server | Steps | Tool Calls | What Claude Does |
|---|---|---|---|
| **DeepWiki** | 1 | 1 | `ask_question("What is the Tensix ISA?")` → receives a clean answer |
| **Glean** | 1-2 | 1-2 | `search("Tensix ISA")` → reads snippets; if insufficient, `chat("What is the Tensix ISA?")` |
| **Atlassian** | 3-6 | 3-6 | `search("Tensix ISA")` → finds page IDs → `getConfluencePage(id1)` → `getConfluencePage(id2)` → overflow handling with `Grep`/`Bash` → Claude synthesizes manually |

### Data Flow Diagrams

The following diagrams trace the full lifecycle of a single user question — from the moment Claude decides to use an MCP server, through every tool call and intermediate step, to the final answer presented to the user. Each diagram annotates what happens at every boundary crossing (LLM → MCP Client → MCP Server → external service) and where context tokens are consumed.

---

#### DeepWiki: Single-Shot RAG

DeepWiki is the simplest flow. The server performs retrieval-augmented generation internally, so Claude sends one question and receives one answer. There is no multi-step orchestration required.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER: "What is the Tensix ISA?"                                              │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM)                                                                 │
│                                                                              │
│  Decision: This is a code/architecture question → use DeepWiki               │
│                                                                              │
│  Emits tool call:                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__deepwiki__ask_question                                        │      │
│  │   repoName: "tenstorrent/tt-metal"                                 │      │
│  │   question: "What is the Tensix ISA?"                              │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT (Claude Code)                                                     │
│                                                                              │
│  1. Strips prefix: mcp__deepwiki__ → server="deepwiki", tool="ask_question"  │
│  2. Forwards request to DeepWiki server via MCP protocol (JSON-RPC)          │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ DEEPWIKI SERVER                                                              │
│                                                                              │
│  Internal RAG pipeline (invisible to Claude):                                │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 1. Embed the question                                               │     │
│  │ 2. Retrieve relevant chunks from tt-metal index (code, docs, READMEs)│    │
│  │ 3. Feed chunks + question to an internal LLM                        │     │
│  │ 4. Generate a grounded, markdown-formatted answer                   │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Returns: clean markdown text (~2-5 KB)                                      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT (Claude Code)                                                     │
│                                                                              │
│  Receives response. Size < 25K tokens → passes inline to Claude              │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM)                                                                 │
│                                                                              │
│  Receives tool result: clean markdown answer                                 │
│  Formats and presents to user.                                               │
│                                                                              │
│  Context consumed: ~2-5 KB (request) + ~2-5 KB (response) = ~4-10 KB total  │
│  Tool calls: 1                                                               │
│  Overflow: No                                                                │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER sees: formatted answer about the Tensix ISA                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- **One tool call, one round-trip.** The server does all the heavy lifting internally.
- **No overflow.** Responses are typically 2-5 KB of clean markdown.
- **No post-processing.** Claude receives ready-to-present text.
- **No authentication overhead.** DeepWiki requires no auth setup calls.

---

#### Glean: Two-Tier Search → Synthesis

Glean has two paths. The **fast path** uses `search` and reads snippets directly. The **slow path** escalates to `chat` for AI synthesis when snippets are insufficient. The slow path is significantly more expensive.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER: "What is the Tensix ISA?"                                              │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM)                                                                 │
│                                                                              │
│  Decision: Enterprise knowledge question → try Glean search first            │
│                                                                              │
│  Emits tool call:                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__glean_default__search                                         │      │
│  │   query: "Tensix ISA"                                              │      │
│  │   app: "confluence"          (optional filter)                     │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT → GLEAN SERVER                                                    │
│                                                                              │
│  Glean indexes search, applies user's permission filter, ranks results.      │
│  Returns: JSON with 10-15 document objects, each containing:                 │
│    - title, url, snippets[]     (~5-10% of payload — the useful part)        │
│    - datasource, datasourceId, documentCategory, matchingFilters,            │
│      owner, updatedBy, parentDocument, superContainerId, similarResults      │
│      (~90-95% of payload — metadata overhead)                                │
│                                                                              │
│  Total response: ~20-40 KB                                                   │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM) — DECISION POINT                                                │
│                                                                              │
│  Reads snippets from search results.                                         │
│                                                                              │
│  ┌─────────────────────┐       ┌──────────────────────────────────────┐      │
│  │ Snippets sufficient? │──YES──│ Format answer from snippets.         │      │
│  │                     │       │ Context: ~20-40 KB total. Done.      │      │
│  └─────────┬───────────┘       └──────────────────────────────────────┘      │
│            NO                                                                │
│            │                                                                 │
│  Escalate to chat:                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__glean_default__chat                                           │      │
│  │   message: "What is the Tensix ISA and what are its key            │      │
│  │             components?"                                           │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │ (slow path only)
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT → GLEAN SERVER                                                    │
│                                                                              │
│  Glean's internal AI pipeline (invisible to Claude):                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 1. Runs multiple internal searches (visible as "Searching..." msgs) │     │
│  │ 2. Reads relevant documents    (visible as "Reading..." msgs)       │     │
│  │ 3. Synthesizes an answer using Glean's AI model                     │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Returns: JSON with messages[] array containing:                             │
│    - UPDATE messages: "Searching...", "Reading..." with structuredResults     │
│      (each structuredResults block = full document metadata envelope)         │
│    - FINAL message: the actual AI answer text (~1-3 KB)                      │
│                                                                              │
│  Total response: ~50-90 KB                                                   │
│  Useful answer: ~1-3 KB (~2-5% of payload)                                   │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT (Claude Code)                                                     │
│                                                                              │
│  Response size > 25K tokens?                                                 │
│  ┌─────┐       ┌────────────────────────────────────────────────────────┐    │
│  │ NO  │───────│ Pass inline to Claude. Claude reads the JSON,          │    │
│  │     │       │ extracts the answer from the final message fragment.   │    │
│  └─────┘       └────────────────────────────────────────────────────────┘    │
│  ┌─────┐       ┌────────────────────────────────────────────────────────┐    │
│  │ YES │───────│ Save to temp file on disk. Return file path to Claude. │    │
│  └─────┘       └────────────────────────────────────────────────────────┘    │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │ (if overflow)
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM) — OVERFLOW HANDLING                                             │
│                                                                              │
│  Emits Bash tool call to extract answer from the overflow file:              │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ python3 -c "                                                       │      │
│  │ import json                                                        │      │
│  │ with open('path/to/result.txt') as f:                              │      │
│  │     data = json.load(f)                                            │      │
│  │ for m in data.get('messages', []):                                 │      │
│  │     for frag in m.get('fragments', []):                            │      │
│  │         t = frag.get('text', '')                                   │      │
│  │         if len(t) > 200:                                           │      │
│  │             print(t)                                               │      │
│  │ "                                                                  │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  Receives extracted answer text (~1-3 KB). Formats and presents to user.     │
│                                                                              │
│  Context consumed (slow path):                                               │
│    search request + response:    ~20-40 KB                                   │
│    chat request:                 ~0.5 KB                                     │
│    chat overflow notice:         ~0.5 KB                                     │
│    Bash extraction command:      ~0.5 KB                                     │
│    Extracted answer:             ~1-3 KB                                     │
│    Total:                        ~25-45 KB                                   │
│  Tool calls: 3 (search + chat + Bash)                                        │
│  Overflow: Yes (chat response)                                               │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER sees: formatted answer about the Tensix ISA                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- **Two paths with very different costs.** The fast path (search only) costs ~20-40 KB. The slow path (search + chat + extraction) costs ~25-45 KB.
- **Metadata bloat is the primary cost driver.** Both `search` and `chat` wrap small amounts of useful content in large enterprise metadata envelopes.
- **Overflow is common on the slow path.** The `chat` response frequently exceeds 25K tokens, requiring a Python-based extraction step.
- **Claude must decide when to escalate.** The fast path vs slow path decision is a judgment call based on snippet quality.

---

#### Atlassian: Multi-Step API Orchestration

The Atlassian flow is the most complex. There is no AI synthesis on the server — Claude must orchestrate multiple raw API calls, handle overflows, and synthesize the answer itself. The diagram below traces the actual flow from the ISA query earlier in this conversation.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER: "Tell me about the Tenstorrent ISA from the Atlassian Confluence"      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM)                                                                 │
│                                                                              │
│  Decision: User wants Confluence content → use Atlassian MCP                 │
│                                                                              │
│  TOOL CALL 1 — Discover relevant content:                                    │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__atlassian__search                                             │      │
│  │   query: "Tenstorrent ISA instruction set architecture"            │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT → ATLASSIAN MCP SERVER → ATLASSIAN CLOUD REST API                 │
│                                                                              │
│  Server translates to: Rovo Search API call                                  │
│  Atlassian returns: 20 results across Confluence pages and Jira issues       │
│                                                                              │
│  Each result contains: id (ARI), title, short text snippet, url, type        │
│  Response: ~3-5 KB (relatively compact — Rovo snippets are brief)            │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM) — ANALYZE SEARCH RESULTS                                        │
│                                                                              │
│  Examines 20 results. Identifies relevant Confluence pages:                  │
│    - "Tensix Instruction Set Architecture" (pageId: 1613201604)              │
│    - "Tensix SFPU Instruction Set Architecture" (pageId: 1170505767)         │
│                                                                              │
│  Snippets are too short to answer fully → need full page content.            │
│  Needs cloudId for subsequent calls.                                         │
│  (Already known from search metadata: b9d94484-5dbd-4ae2-b670-6f414aefb4cd) │
│                                                                              │
│  TOOL CALLS 2 & 3 — Fetch both pages in parallel:                           │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__atlassian__getConfluencePage                                  │      │
│  │   cloudId: "b9d94484-5dbd-4ae2-b670-6f414aefb4cd"                 │      │
│  │   pageId: "1613201604"                                             │      │
│  │   contentFormat: "markdown"                                        │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ mcp__atlassian__getConfluencePage                                  │      │
│  │   cloudId: "b9d94484-5dbd-4ae2-b670-6f414aefb4cd"                 │      │
│  │   pageId: "1170505767"                                             │      │
│  │   contentFormat: "markdown"                                        │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────┬──────────────────────────────────────┬────────────────────────────┘
           │                                      │
           ▼                                      ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│ ATLASSIAN → Page 1 (ISA)        │  │ ATLASSIAN → Page 2 (SFPU ISA)           │
│                                 │  │                                         │
│ Returns: JSON with full page    │  │ Returns: JSON with full page             │
│   id, title, spaceId, version,  │  │   Same structure but body is 175,548    │
│   body (markdown)               │  │   characters long                        │
│                                 │  │                                         │
│ Response: ~4 KB                 │  │ Response: ~175 KB                        │
│ Fits inline: YES                │  │ Fits inline: NO → OVERFLOW TO FILE      │
└──────────┬──────────────────────┘  └──────────┬──────────────────────────────┘
           │                                    │
           ▼                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MCP CLIENT (Claude Code)                                                     │
│                                                                              │
│  Page 1: Passes inline to Claude (fits in context)                           │
│  Page 2: Saves to temp file, returns error message with file path:           │
│                                                                              │
│    "Error: result (175,548 characters) exceeds maximum allowed tokens.       │
│     Output has been saved to:                                                │
│     /home/user/.claude/projects/.../tool-results/                            │
│       mcp-atlassian-getConfluencePage-1772636372291.txt"                     │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM) — OVERFLOW HANDLING                                             │
│                                                                              │
│  Has Page 1 content inline (the ISA instruction listing).                    │
│  Page 2 overflowed. Must extract relevant content from the file.             │
│                                                                              │
│  TOOL CALL 4 — Search overflow file for key sections:                        │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ Grep                                                               │      │
│  │   pattern: "Overview|SFPU|register file|lane|instruction format"   │      │
│  │   path: "/home/user/.claude/.../mcp-atlassian-...-1772636372291.txt│      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  Result: Grep returns matching lines, but file is JSON — line-oriented       │
│  search is not ideal for structured content.                                 │
│                                                                              │
│  TOOL CALL 5 — Extract beginning of page body using Python:                  │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ Bash: python3 -c "                                                 │      │
│  │   import json                                                      │      │
│  │   with open('path/to/result.txt') as f:                            │      │
│  │       data = json.load(f)                                          │      │
│  │   for item in data:                                                │      │
│  │       text = item.get('text', '')                                  │      │
│  │       print(text[:5000])                                           │      │
│  │ "                                                                  │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  Returns: first 5000 chars of the SFPU ISA page (Introduction, Feature       │
│  Summary, Limitations, Compute Organization, Architectural State...)         │
│                                                                              │
│  TOOL CALL 6 — Extract more content from further into the document:          │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │ Bash: python3 -c "                                                 │      │
│  │   ...                                                              │      │
│  │   print(text[5000:10000])                                          │      │
│  │ "                                                                  │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  Returns: next 5000 chars (Constant Registers, Control Registers...)         │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CLAUDE (LLM) — MANUAL SYNTHESIS                                              │
│                                                                              │
│  Now has:                                                                    │
│    - Page 1 inline: Full Tensix ISA instruction listing (~4 KB)              │
│    - Page 2 extracted: SFPU introduction, features, arch state (~10 KB)      │
│                                                                              │
│  Claude must now:                                                            │
│    1. Read and understand both pages                                         │
│    2. Identify the most relevant information                                 │
│    3. Synthesize a coherent answer                                           │
│    4. Format and present to the user                                         │
│                                                                              │
│  (This is work that DeepWiki/Glean servers do internally.)                   │
│                                                                              │
│  Context consumed:                                                           │
│    search request + response:           ~4 KB                                │
│    getConfluencePage (page 1) req+resp: ~5 KB                                │
│    getConfluencePage (page 2) overflow: ~1 KB (error message)                │
│    Grep request + response:             ~1 KB                                │
│    Bash extraction 1 request + output:  ~6 KB                                │
│    Bash extraction 2 request + output:  ~6 KB                                │
│    Total:                               ~23 KB                               │
│  Tool calls: 6                                                               │
│  Overflow: Yes (page 2)                                                      │
└──────────┬───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ USER sees: Claude's manually synthesized answer about the Tensix ISA         │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- **Multiple sequential tool calls.** Each call depends on the previous one's results (search → page IDs → fetch → overflow path → extraction).
- **Claude is the synthesizer.** Unlike DeepWiki and Glean, there is no AI on the server side. Claude must read raw content and produce the answer.
- **Overflow is the norm, not the exception.** Any Confluence page longer than a few thousand words will overflow.
- **Extraction is iterative.** Claude may need multiple Bash calls to read different portions of a large overflow file, each consuming additional context.
- **The cloudId tax.** Every Atlassian call carries a mandatory `cloudId` parameter, and the first call in a session must obtain it.

---

#### Side-by-Side Flow Comparison

```
                 DeepWiki              Glean (slow path)         Atlassian
                 ────────              ─────────────────         ─────────
Step 1           ask_question          search                   search (Rovo)
                     │                     │                        │
                     │                     ▼                        ▼
Step 2               │                 chat                    getConfluencePage ×2
                     │                     │                   (parallel)
                     │                     │                        │
                     │                     ▼                        ▼
Step 3               │                 Bash (extract           Grep (search overflow)
                     │                  from overflow)              │
                     │                     │                        ▼
Step 4               │                     │                   Bash (extract chunk 1)
                     │                     │                        │
                     │                     │                        ▼
Step 5               │                     │                   Bash (extract chunk 2)
                     │                     │                        │
                     ▼                     ▼                        ▼
Result           Server-generated      Server-generated        Claude-generated
                 answer                answer (extracted)      answer (synthesized)
                 ─────────────────────────────────────────────────────────────
Tool calls:      1                     3                       6
Context:         ~4-10 KB              ~25-45 KB               ~23 KB
Overflow:        No                    Yes (chat)              Yes (large page)
Synthesis by:    DeepWiki server       Glean server            Claude (LLM)
```

---

## 4. Context Cost Comparison

This is the most important practical difference between the servers.

### Per-Operation Cost

| Operation | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| Search / discovery | ~2-5 KB | ~20-40 KB | ~3-5 KB (Rovo), ~5-15 KB (CQL/JQL) |
| Full content retrieval | ~10-50 KB | Varies by doc | ~4 KB - 175 KB+ (raw page content) |
| Question answering | ~2-5 KB | ~50-90 KB | N/A (no synthesis tool) |
| **Overhead ratio** | **~5-10%** | **~90-98%** | **~20-50%** (varies widely) |

### Why Each Server Costs What It Does

| Server | Primary Cost Driver |
|---|---|
| **DeepWiki** | Minimal — server does RAG internally, returns clean text. Almost all returned tokens are useful content. |
| **Glean** | Enterprise metadata bloat — each result carries ~2-4 KB of metadata (datasource IDs, container IDs, owner info, similar results). The `chat` tool returns ~85 KB of JSON for a ~1.5 KB answer. |
| **Atlassian** | Raw API verbosity + overflow handling — Confluence pages can be 175 KB+ in a single response. When they overflow to disk, additional `Read`/`Grep`/`Bash` calls pile on more context. No server-side filtering or summarization. |

### Concrete Example: "Tell Me About the Tenstorrent ISA"

| Metric | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| Tool calls needed | 1 | 1-2 | 6 |
| Total response size | ~3-5 KB | ~50-90 KB | ~180+ KB (before overflow handling) |
| Useful content in response | ~95% | ~2-5% | ~10-30% |
| Overflow to file? | No | Often (`chat`) | Often (large Confluence pages) |
| Secondary extraction needed? | No | Yes (Python JSON parsing) | Yes (Grep, Bash, Read with offset) |
| Estimated context consumed | ~5 KB | ~15-25 KB | ~15-20 KB (after extraction) |

### Overflow Behavior

All three servers can trigger Claude Code's 25K-token overflow limit, but the frequency and handling differ:

| Server | Overflow Frequency | What Overflows | Recovery Method |
|---|---|---|---|
| DeepWiki | Rare | `read_wiki_contents` (full repo docs) | `Read` with offset/limit |
| Glean | Common | `chat` responses (metadata bloat) | `Bash` with Python JSON extraction |
| Atlassian | Common | Large Confluence pages (175 KB+) | `Read`/`Grep`/`Bash` on saved file |

---

## 5. Query Language Comparison

Each server accepts queries differently:

| Aspect | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| **Query style** | Natural language | Keywords + filters | Keywords (Rovo), CQL (Confluence), JQL (Jira) |
| **Natural language support** | Full (`ask_question`) | Full (`chat`), keywords only (`search`) | None — keyword or structured query only |
| **Filter capabilities** | None (repo-scoped) | Rich: `app`, `from`, `owner`, `updated`, `type`, `channel`, dynamic filters | CQL: `space`, `title`, `type`, `label`, `ancestor`; JQL: `project`, `status`, `assignee`, `priority`, dates, custom fields |
| **Query languages** | N/A | N/A | CQL (Confluence Query Language), JQL (Jira Query Language) |
| **Cross-source search** | Cross-repo (up to 10) | Cross-app (all indexed sources) | Cross-product (Jira + Confluence via Rovo `search`) |

### Query Examples for the Same Intent

**"Find ISA-related content":**

```
DeepWiki:  ask_question(repoName: "tenstorrent/tt-metal", question: "What is the Tensix ISA?")
Glean:     search(query: "Tensix ISA", app: "confluence")
Atlassian: searchConfluenceUsingCql(cloudId: "...", cql: "title ~ \"Tensix ISA\" AND type = page AND space = TA")
```

**"Find recent Jira issues about SFPU":**

```
DeepWiki:  (cannot — no Jira access)
Glean:     search(query: "SFPU", app: "jira", updated: "past_week")
Atlassian: searchJiraIssuesUsingJql(cloudId: "...", jql: "text ~ \"SFPU\" AND updated >= -7d", fields: ["summary","status"])
```

---

## 6. Write Capabilities

Only the Atlassian server supports write operations.

| Operation | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| Create wiki page | - | - | `createConfluencePage` |
| Update wiki page | - | - | `updateConfluencePage` |
| Create Jira issue | - | - | `createJiraIssue` |
| Update Jira issue | - | - | `editJiraIssue` |
| Transition issue status | - | - | `transitionJiraIssue` |
| Add comments | - | - | `createConfluenceFooterComment`, `createConfluenceInlineComment`, `addCommentToJiraIssue` |
| Log work | - | - | `addWorklogToJiraIssue` |
| Link issues | - | - | `jiraWrite(action: "createIssueLink")` |

This makes the Atlassian server the only bidirectional integration — Claude can not only read but also create and modify enterprise content.

---

## 7. Response Format Comparison

| Aspect | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| **Format** | Plain markdown text | JSON with metadata envelope | JSON (raw Atlassian API response) |
| **Content-to-metadata ratio** | ~95% content | ~2-5% content | ~10-80% content (depends on endpoint) |
| **Needs post-processing** | No | Often (JSON extraction for `chat`) | Often (overflow handling, field extraction) |
| **Human-readable as-is** | Yes | No (JSON) | Partially (markdown body embedded in JSON) |

### Example Responses for a Confluence Page

**DeepWiki** (same content from the GitHub repo):
```markdown
The Tensix ISA defines instructions for the Tensix compute core...
[clean, readable markdown]
```

**Glean** (via search):
```json
{
  "documents": [{
    "title": "Tensix ISA",
    "url": "https://...",
    "snippets": ["The SFPU is a SIMD engine..."],
    "datasource": "confluence",
    "datasourceId": "...",
    "documentCategory": "...",
    "matchingFilters": { ... },
    "owner": { ... },
    ...20+ more metadata fields
  }]
}
```

**Atlassian** (via getConfluencePage):
```json
{
  "id": "1613201604",
  "type": "page",
  "status": "current",
  "title": "Tensix Instruction Set Architecture",
  "spaceId": "31293488",
  "version": { "number": 2, "authorId": "...", "createdAt": "..." },
  "body": "## Overview\n\nA brief summary of the Quasar instruction set..."
}
```

The Atlassian response is the most predictable — it mirrors the REST API structure. The `body` field contains the useful content. Glean wraps minimal content in maximal metadata. DeepWiki returns content directly.

---

## 8. Data Source Overlap

The three servers partially overlap on Confluence and Jira:

```
                    ┌─────────────────────────────────────────────┐
                    │                  Glean                       │
                    │  SharePoint, Slack, Google Drive, Gmail,     │
                    │  Greenhouse, Airtable, Tableau, Notion,     │
                    │  Datadog, ...                                │
                    │  ┌───────────────────────────────────┐      │
                    │  │         Atlassian MCP              │      │
                    │  │   Confluence (read + write)        │      │
                    │  │   Jira (read + write)              │      │
                    │  │   Comments, Transitions, Worklogs  │      │
                    │  └───────────────────────────────────┘      │
                    │  Confluence (read-only, indexed snapshots)   │
                    │  Jira (read-only, indexed snapshots)         │
                    └─────────────────────────────────────────────┘

    ┌───────────────────────┐
    │       DeepWiki         │
    │  GitHub repositories   │
    │  (code + docs)         │
    └───────────────────────┘
```

Key distinctions:
- **Glean indexes Confluence and Jira** but provides read-only access to indexed snapshots. It may not reflect the latest edits.
- **Atlassian provides real-time access** to Confluence and Jira, including the latest content, plus full write capability.
- **DeepWiki covers GitHub** exclusively, with no overlap with the other two on Confluence/Jira data.

### When the Same Content Exists in Multiple Servers

For Confluence content that also lives in a GitHub repo (e.g., architecture docs mirrored to both):

| Access Via | Freshness | Context Cost | Write Access |
|---|---|---|---|
| DeepWiki | Depends on indexing lag | Low (~2-5 KB) | No |
| Glean | Depends on indexing lag | High (~20-40 KB) | No |
| Atlassian | Real-time | Variable (4 KB - 175 KB+) | Yes |

---

## 9. Decision Matrix: Which Server to Use

### By Information Type

| You need... | Use | Why |
|---|---|---|
| Code-level understanding of a GitHub repo | **DeepWiki** | Purpose-built for code Q&A, lowest context cost |
| An internal SharePoint document | **Glean** | Only server indexing SharePoint |
| A Slack message or email thread | **Glean** | Only server indexing Slack/email |
| The latest version of a Confluence page | **Atlassian** | Real-time access, not dependent on indexing |
| A Jira issue's full details | **Atlassian** | Real-time, structured access with field filtering |
| Cross-source synthesis ("what do we know about X?") | **Glean** | Searches all indexed enterprise sources at once |
| To create or update a Confluence page | **Atlassian** | Only server with write capabilities |
| To create or transition a Jira issue | **Atlassian** | Only server with write capabilities |

### By Priority (Context Budget)

When multiple servers could answer the question, prefer them in this order:

1. **DeepWiki** — Cheapest context cost, clean answers, no auth overhead. Use whenever the answer might exist in a GitHub repository.
2. **Glean `search`** — Moderate cost. Use for enterprise knowledge discovery. Read snippets before deciding to go deeper.
3. **Atlassian `search` (Rovo)** — Similar cost to Glean search. Use when you need Confluence/Jira-specific results.
4. **Glean `read_document`** — Variable cost. Use to read a specific document found via search.
5. **Atlassian `getConfluencePage` / `getJiraIssue`** — Variable cost, risk of overflow. Use for real-time access or when Glean's indexed version is stale.
6. **Glean `chat`** — High cost. Use only when cross-source synthesis is genuinely required.

### By Workflow Type

| Workflow | Recommended Approach |
|---|---|
| **Quick factual question** | DeepWiki `ask_question` (if code-related) or Glean `search` (if enterprise) |
| **Find a specific document** | Glean `search` → read snippets → `read_document` if needed |
| **Read a Confluence page in full** | Atlassian `getConfluencePage` (real-time) or Glean `read_document` (indexed) |
| **Search Jira issues** | Atlassian `searchJiraIssuesUsingJql` (precise JQL) or Glean `search` (keyword) |
| **Create/update content** | Atlassian (only option) |
| **Cross-source research** | Glean `search` (broad), then targeted follow-ups with Atlassian or DeepWiki |
| **Architecture deep-dive** | DeepWiki `ask_question` for code-level, Atlassian for Confluence specs |

---

## 10. Summary Table

| Dimension | DeepWiki | Glean | Atlassian |
|---|---|---|---|
| **Design** | RAG synthesis engine | Enterprise search + AI chat | REST API proxy |
| **Tools** | 3 | 3 | 30+ |
| **Data sources** | GitHub repos | All enterprise tools | Confluence + Jira |
| **Query style** | Natural language | Keywords + filters / natural language | Keywords / CQL / JQL |
| **Response format** | Clean markdown | JSON with heavy metadata | Raw API JSON |
| **Content ratio** | ~95% | ~2-5% | ~10-80% |
| **Avg. context cost** | Low (2-5 KB) | High (20-90 KB) | Variable (3-175 KB+) |
| **Overflow frequency** | Rare | Common (`chat`) | Common (large pages) |
| **Write operations** | No | No | Yes (full CRUD) |
| **Auth required** | No | Yes | Yes |
| **Best for** | Code Q&A | Cross-source enterprise search | Real-time Confluence/Jira access + write ops |
| **Worst for** | Non-code enterprise content | Context-constrained conversations | Simple questions (high overhead) |
