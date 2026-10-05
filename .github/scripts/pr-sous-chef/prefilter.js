'use strict';

// Pre-activation prefilter for .github/workflows/pr-sous-chef.md (gh-aw). Runs in an
// `actions/github-script` step of the pre_activation job with GITHUB_TOKEN; the .md
// requires this file after a sparse checkout and calls `run({ github, context, core })`.
// It lives here rather than inline because GitHub Actions caps a step input at 21 KB
// (gh-aw enforces it at compile time) and so that the decision predicates can be unit
// tested (prefilter.test.js) against the real gh-aw output sanitizer.
//
// Deterministic port of upstream github/gh-aw pr-sous-chef.md's `fetch-prs` bash step.
// Everything the agent later needs is gathered here, so the agent job can run with only
// the `pull_requests` + `actions` toolsets and no bash. Fails closed per PR: a PR whose
// state could not be read is skipped this tick.

// ---- tt-metal configuration ---------------------------------------------------------
// Provenance label applied by copilot-flow-ready.yaml (see that file's header).
const FLOW_LABEL = 'copilot-flow';
// Applied by THIS script (best effort) together with the hand-off comment when a PR hits
// the per-PR nudge cap. Human-visible, removable, and it keeps handed-off PRs out of the
// search query. Its REMOVAL is the reset: the timestamp of the most recent `unlabeled`
// event for this label on the PR is the baseline from which nudges are counted again
// (see `nudgeBaseline` in run()).
const HANDOFF_LABEL = 'copilot-flow-handoff';
// Marker of the hand-off comment. It is written by this script directly through the REST
// API, NOT through a safe output, because gh-aw's output sanitizer strips every HTML
// comment from agent-authored content before posting (gh-aw-actions
// setup/js/sanitize_content.cjs, `removeXmlComments`; verified against the pinned commit
// 6aab9e5b the workflow runs). A marker the agent was asked to emit never reached GitHub.
// Trusted only in comments authored by github-actions[bot], the identity the
// pre_activation job's GITHUB_TOKEN writes as.
const HANDOFF_MARKER = '<!-- gh-aw-pr-sous-chef-handoff -->';
// How the workflow recognises its OWN nudges (cap, cooldown, "no two in a row"): gh-aw's
// add-comment handler appends, AFTER sanitization, an XML marker of the form
// `<!-- gh-aw-agentic-workflow: <name>, ..., workflow_id: pr-sous-chef, run: <url> -->`
// (gh-aw-actions setup/js/generate_footer.cjs). The sanitizer removes any HTML comment
// from the agent's own text, so the agent cannot forge this marker, and a comment carrying
// it together with an `@copilot` mention is an actionable nudge; one carrying it WITHOUT
// the mention is informational (the `/souschef` reply). Must equal the compiled
// GH_AW_WORKFLOW_ID, i.e. the workflow file's basename.
const WORKFLOW_ID = 'pr-sous-chef';
// Upstream: 30 min. A tt-metal Copilot session runs up to 59 min and pr-gate ~20 min, and
// the session-in-progress filter already covers "Copilot is on it"; 60 min is for the case
// where the mention did not start a session at all.
const COOLDOWN_MS = 60 * 60 * 1000;
// Upstream ignores checks that have been running for >1h so long agentic checks do not
// block nudges forever. pr-gate on a Copilot PR takes ~20 min, but tt-metal's hardware
// queues can hold a job QUEUED much longer; 90 min keeps nudges from firing during a
// merely slow (not stuck) pipeline.
const PENDING_CHECK_MAX_AGE_MS = 90 * 60 * 1000;
// NEW vs upstream: age past which an unmatched `copilot_work_started` (no later `finished`
// or `finished_failure`) is treated as stale rather than "still running" (see Filter 2).
// Same 90 min reasoning as PENDING_CHECK_MAX_AGE_MS above it: sessions run up to 59 min per
// the COOLDOWN_MS comment, so 90 min is well past a genuine one but still bounded.
const SESSION_STALE_MS = 90 * 60 * 1000;
// Upstream default: a PR with zero changed files 24h after creation is "stalled".
const ZERO_DIFF_AGE_MS = 24 * 60 * 60 * 1000;
// NEW vs upstream: after this many actionable nudges on one PR, stop mentioning Copilot
// and post one hand-off comment for humans instead. Without a cap, a PR whose CI keeps
// failing would burn one Copilot session per hour indefinitely.
const MAX_NUDGES_PER_PR = 6;
// Upstream: 4 nudges per run. Also the hard cap on how many PRs' context goes into the
// agent prompt (each PR ~2-4 KB with thread excerpts; the step-output limit is 1 MB but
// every excerpt is untrusted text in the prompt), and on hand-offs per run.
const MAX_ELIGIBLE = 4;
// Untrusted review-comment text placed in the agent prompt is truncated to this.
const EXCERPT_MAX = 300;
// Unanswered threads per PR carried into the PROMPT (oldest first, so the longest-waiting
// thread is always the first one surfaced). Token-budget limit only: whether a PR is
// actionable, and which bot threads are resolvable, is decided over the COMPLETE,
// paginated thread list before any truncation.
const MAX_THREADS_PER_PR = 10;
// Combined cap on thread IDs handed to the agent for resolution per run. MUST equal
// `safe-outputs.resolve-pull-request-review-thread.max` in the .md: the prompt says
// "resolve every listed ID", so the list has to fit the handler's budget or the last calls
// of a large backlog would be rejected nondeterministically. IDs beyond the cap are simply
// deferred to the next tick.
const RESOLVE_THREADS_MAX = 20;
// Review threads are paginated 100 at a time; refuse to decide on more than this many
// pages rather than on a partial list.
const MAX_THREAD_PAGES = 20;
// The Copilot coding agent's logins: `copilot-swe-agent` (GraphQL) and `Copilot`
// (REST / assignee). EXACT match, deliberately not a prefix: `copilot-pull-request-
// reviewer` (Copilot code review) also starts with "copilot" and is a different bot whose
// follow-up must never count as "Copilot replied".
const COPILOT_CODING_AGENT_LOGINS = new Set(['copilot-swe-agent', 'copilot']);
// Explicit allowlist of review-bot logins whose unresolved threads may be resolved once
// Copilot's reply verifiably points at a fix (see `verifyFix`). Case-normalized, `[bot]`
// suffix stripped, and the actor must be GraphQL type Bot. Verified against real Copilot
// PRs in this repo (2026-09-28):
//   copilot-pull-request-reviewer  Copilot code review (ruleset "Automatic Copilot PR Review")
//   github-actions                 the gh-aw reviewers (tenstorrent-skills-reviewer,
//                                  mattpocock-skills-reviewer) post their review threads
//                                  with GITHUB_TOKEN, so this is their GraphQL login; the
//                                  repo has no GH_AW_GITHUB_TOKEN secret that would change
//                                  it. Any other repo workflow posting review threads as
//                                  github-actions is repo-controlled code.
// NOT on the list, and seen on Copilot PRs here: `cycode-security` (Bot). Its threads —
// and any other bot's — are treated exactly like a human reviewer's: listed for Copilot to
// answer, never resolved by this workflow.
const RESOLVABLE_REVIEWER_BOTS = new Set(['copilot-pull-request-reviewer', 'github-actions']);
// A Copilot reply that says it could NOT do something vetoes auto-resolution even when it
// also cites a commit. Secondary guard only; the primary evidence is the cited commit.
const REPLY_VETO = /\b(can(?:no|')t|could ?not|couldn't|unable|not (?:able|possible)|(?:needs?|requires?) (?:a )?(?:human|maintainer)|out of scope|won't|will not|did ?not|didn't|blocked|not (?:yet )?(?:addressed|fixed|done|implemented))\b/i;
// The hand-off comment is written by the pre_activation job's GITHUB_TOKEN, whatever the
// nudges use.
const PRE_ACTIVATION_LOGIN = 'github-actions[bot]';
// --------------------------------------------------------------------------------------

// Same predicate as gh-aw-actions generate_footer.cjs `matchesWorkflowId` at the pinned commit.
const matchesWorkflowId = (body, workflowId = WORKFLOW_ID) => {
  if (!body || body.includes('<!-- gh-aw-comment-type: reaction -->')) return false;
  if (body.includes(`<!-- gh-aw-workflow-id: ${workflowId} -->`)) return true;
  return body.includes('<!-- gh-aw-agentic-workflow:')
    && (body.includes(`workflow_id: ${workflowId},`) || body.includes(`workflow_id: ${workflowId} -->`));
};
const isCopilotCodingAgent = (login) => COPILOT_CODING_AGENT_LOGINS.has((login || '').toLowerCase());
const isResolvableReviewerBot = (node) => node?.__typename === 'Bot'
  && RESOLVABLE_REVIEWER_BOTS.has((node?.login || '').toLowerCase().replace(/\[bot\]$/, ''));
// `MergeStateStatus` has no CONFLICTING value (live schema: DIRTY, UNKNOWN, BLOCKED, BEHIND,
// UNSTABLE, HAS_HOOKS, CLEAN); a conflicting PR reports mergeStateStatus DIRTY and
// mergeable CONFLICTING (both checked live on real conflicting tt-metal PRs, 2026-09-28).
// Either is accepted; UNKNOWN (GitHub still computing) counts as not conflicting this tick.
const isConflicting = (pr) => pr?.mergeStateStatus === 'DIRTY' || pr?.mergeable === 'CONFLICTING';
// Identity predicates. `sousChefLogin` is the login the add-comment safe output posts under
// (resolved by the workflow's `identity` step: GH_AW_AGENT_TOKEN's user when provisioned,
// github-actions[bot] otherwise). Only comments from that login that also carry the
// handler's post-sanitization marker count as the workflow's own.
const makeIdentity = (sousChefLogin) => {
  const login = (sousChefLogin || PRE_ACTIVATION_LOGIN).toLowerCase();
  const isOwnComment = (c) => (c?.user?.login || '').toLowerCase() === login && matchesWorkflowId(c?.body);
  const isNudge = (c) => isOwnComment(c) && (c.body || '').includes('@copilot');
  const isPreActivationComment = (c) => c?.user?.type === 'Bot' && (c?.user?.login || '').toLowerCase() === PRE_ACTIVATION_LOGIN;
  const isHandoff = (c) => isPreActivationComment(c) && (c.body || '').includes(HANDOFF_MARKER);
  return { login, isOwnComment, isNudge, isHandoff };
};
// A reply alone is NOT evidence that a review concern was addressed ("I could not fix this"
// is a reply too). A bot thread is resolvable only when Copilot's reply cites a commit SHA
// that is a commit of the PR made after the thread was opened — Copilot's fix replies here
// consistently read "Fixed in commit 96fa8f3" / "(df3d528)" — and the reply contains no
// "could not / needs a maintainer" phrasing. `prCommits` = [{ sha, at }] (lowercase sha,
// epoch ms); `openedAt` = epoch ms of the thread's first comment.
const verifyFix = (replyBody, prCommits, openedAt) => {
  const body = replyBody || '';
  if (REPLY_VETO.test(body)) return false;
  const cited = [...new Set(body.toLowerCase().match(/\b[0-9a-f]{7,40}\b/g) || [])];
  return cited.some(h => (prCommits || []).some(c => c.sha.startsWith(h) && c.at > openedAt));
};
const excerpt = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, EXCERPT_MAX);

async function run({ github, context, core }) {
  const { owner, repo } = context.repo;
  const now = Date.now();
  const identity = makeIdentity(process.env.SOUS_CHEF_LOGIN);
  const { isNudge, isHandoff } = identity;
  const counters = {
    fetched: 0, filtered_flow_label_missing: 0, filtered_checks_pending: 0, filtered_copilot_session_active: 0,
    filtered_handed_off: 0, handed_off_now: 0, handoff_skipped_stale_state: 0, filtered_last_comment_from_sous_chef: 0,
    filtered_cooldown: 0, filtered_copilot_unassigned: 0, filtered_nothing_actionable: 0,
    filtered_error: 0, eligible: 0, resolvable_threads_deferred: 0,
    bot_threads_answered_but_unverified: 0
  };
  const reasons = {}; // number -> why it was filtered (for the /souschef reply)
  const eligible = [];
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;
  let handoffsThisRun = 0;

  const q = `repo:${owner}/${repo} is:pr is:open draft:false label:${FLOW_LABEL} -label:${HANDOFF_LABEL} author:app/copilot-swe-agent`;
  const search = await github.rest.search.issuesAndPullRequests({ q, per_page: 50, sort: 'updated', order: 'desc' });
  counters.fetched = search.data.items.length;
  core.info(`Candidates: ${counters.fetched}`);

  for (const item of search.data.items) {
    const number = item.number;
    try {
      // One GraphQL query per PR for everything except review threads and the Copilot
      // session events. statusCheckRollup contexts are paginated: pr-gate alone fans out
      // into dozens of check runs.
      const fetchPr = (after) => github.graphql(`
        query($owner: String!, $repo: String!, $number: Int!, $after: String) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              number title url isDraft state createdAt updatedAt changedFiles
              mergeStateStatus mergeable reviewDecision headRefOid headRefName
              author { login }
              assignees(first: 20) { nodes { login } }
              labels(first: 50) { nodes { name } }
              commits(last: 1) {
                nodes { commit { committedDate
                  statusCheckRollup { contexts(first: 100, after: $after) {
                    pageInfo { hasNextPage endCursor }
                    nodes {
                      __typename
                      ... on CheckRun { name status conclusion startedAt detailsUrl }
                      ... on StatusContext { context state targetUrl }
                    } } } } }
              }
            }
          }
        }`, { owner, repo, number, after });
      let data = await fetchPr(null);
      const pr = data?.repository?.pullRequest;
      if (!pr || pr.state !== 'OPEN' || pr.isDraft) { reasons[number] = 'no longer open and non-draft'; continue; }
      let contexts = pr.commits?.nodes?.[0]?.commit?.statusCheckRollup?.contexts;
      const checks = [...(contexts?.nodes || [])];
      let guard = 0;
      while (contexts?.pageInfo?.hasNextPage && guard++ < 10) {
        data = await fetchPr(contexts.pageInfo.endCursor);
        contexts = data?.repository?.pullRequest?.commits?.nodes?.[0]?.commit?.statusCheckRollup?.contexts;
        checks.push(...(contexts?.nodes || []));
      }
      const labels = (pr.labels?.nodes || []).map(l => (l.name || '').toLowerCase());
      // Filter 0 — opt-in label. The search index that produced the candidate list can lag
      // behind the PR's real labels; the GraphQL read above is authoritative for THIS tick.
      // A PR without `copilot-flow` is not ours (a maintainer removed it = kill switch).
      if (!labels.includes(FLOW_LABEL)) { counters.filtered_flow_label_missing++; reasons[number] = `not labeled ${FLOW_LABEL} (search index was stale, or a maintainer opted the PR out)`; continue; }

      // Filter 1 — checks still running (young enough to be genuinely in flight).
      const pendingCutoff = now - PENDING_CHECK_MAX_AGE_MS;
      const checksPending = checks.some(c => {
        if (c.__typename === 'CheckRun') {
          const pending = ['QUEUED', 'IN_PROGRESS', 'REQUESTED', 'PENDING'].includes(c.status || 'COMPLETED');
          if (!pending) return false;
          const ts = c.startedAt ? new Date(c.startedAt).getTime() : null;
          return ts === null || ts > pendingCutoff;
        }
        return c.__typename === 'StatusContext' && c.state === 'PENDING';
      });
      if (checksPending) { counters.filtered_checks_pending++; reasons[number] = 'checks still running'; continue; }

      // Filter 2 (new vs upstream) — Copilot is working on it right now. Exact signal: the
      // latest copilot_work_* timeline event is copilot_work_started.
      const timeline = await github.paginate(github.rest.issues.listEventsForTimeline, { owner, repo, issue_number: number, per_page: 100 });
      let session = 'none';
      let sessionAt = null;
      let lastStartedAt = null;
      // Nudge-count baseline (the documented RESET): a maintainer puts a handed-off PR back
      // under Sous Chef by removing `copilot-flow-handoff`. Only nudges and hand-off comments
      // posted AFTER the most recent such removal count; everything before it is history.
      // Without this baseline the old nudge comments (which nobody deletes) would still
      // satisfy the cap and the very next eligible tick would hand the PR off again. Same
      // timeline read copilot-flow-ready.yaml uses for its own sticky opt-out.
      let nudgeBaseline = 0;
      for (const ev of timeline) {
        if (typeof ev.event !== 'string') continue;
        if (ev.event.startsWith('copilot_work_')) {
          session = ev.event; sessionAt = ev.created_at || null;
          if (ev.event === 'copilot_work_started') lastStartedAt = ev.created_at || null;
        }
        if (ev.event === 'unlabeled' && (ev.label?.name || '').toLowerCase() === HANDOFF_LABEL) {
          nudgeBaseline = Math.max(nudgeBaseline, new Date(ev.created_at || 0).getTime());
        }
      }
      // A `copilot_work_started` with no later `copilot_work_finished*` normally means the
      // session is genuinely running (comment above: up to 59 min). But the finished event is
      // occasionally never emitted at all (observed on PR #58341: started 05:23, no commit or
      // finished event for 2h+) and this filter has no other way to notice — unlike Filter 1's
      // checks-pending, which self-heals once a check actually completes. Without an age cutoff
      // a dropped finished event blocks the PR forever. SESSION_STALE_MS is deliberately looser
      // than PENDING_CHECK_MAX_AGE_MS: a real session can legitimately run close to 59 min, and
      // being slow to unstick a merely-slow one is much cheaper than nudging Copilot mid-session.
      if (session === 'copilot_work_started') {
        const startedAt = sessionAt ? new Date(sessionAt).getTime() : null;
        const stale = startedAt !== null && now - startedAt >= SESSION_STALE_MS;
        if (!stale) { counters.filtered_copilot_session_active++; reasons[number] = 'Copilot session in progress'; continue; }
        counters.session_stale_override = (counters.session_stale_override || 0) + 1;
        core.info(`#${number}: copilot_work_started at ${sessionAt} has no finished event after ${SESSION_STALE_MS / 60000} min; treating the session as stale, not blocking`);
      }

      // ALL issue comments (REST, paginated, newest first). They are the persistence for
      // the nudge cap, the cooldown and the hand-off marker, so a recency window is not
      // acceptable: on a busy PR thirty newer comments would hide the history and restart
      // a fresh nudge cycle.
      const comments = (await github.paginate(github.rest.issues.listComments, { owner, repo, issue_number: number, per_page: 100 }))
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const afterBaseline = (c) => new Date(c.created_at).getTime() > nudgeBaseline;

      // Memoized: the silent-session check below (Filter 4) may need review threads before an
      // eligible PR otherwise would, but must not fetch them twice. Cursor-paginated -- a PR
      // with more threads than one page must not silently lose the older ones: an old,
      // still-unanswered thread is exactly what a nudge exists for.
      let threadNodesCache = null;
      const fetchThreadNodes = async () => {
        if (threadNodesCache) return threadNodesCache;
        const nodes = [];
        let cursor = null;
        let pageGuard = 0;
        do {
          const res = await github.graphql(`
            query($owner: String!, $repo: String!, $number: Int!, $after: String) {
              repository(owner: $owner, name: $repo) {
                pullRequest(number: $number) {
                  reviewThreads(first: 100, after: $after) {
                    pageInfo { hasNextPage endCursor }
                    nodes { id isResolved isOutdated path
                      comments { totalCount }
                      firstComment: comments(first: 1) { nodes { author { login __typename } body createdAt url } }
                      lastComment: comments(last: 1) { nodes { author { login __typename } body createdAt } } }
                  }
                }
              }
            }`, { owner, repo, number, after: cursor });
          const conn = res?.repository?.pullRequest?.reviewThreads;
          if (!conn) throw new Error('GraphQL response has no reviewThreads connection');
          nodes.push(...(conn.nodes || []));
          cursor = conn.pageInfo?.hasNextPage ? conn.pageInfo.endCursor : null;
        } while (cursor && pageGuard++ < MAX_THREAD_PAGES);
        if (cursor) throw new Error(`review threads exceed ${MAX_THREAD_PAGES * 100}; refusing to decide on a partial list`);
        threadNodesCache = nodes;
        return nodes;
      };
      // Filter 3 (new) — handed off to humans after the nudge cap: label OR trusted marker
      // comment newer than the baseline (either alone is enough; both are written together
      // below). A hand-off comment from BEFORE the label was removed is the previous cycle's
      // explanation and may stay on the PR as history.
      if (labels.includes(HANDOFF_LABEL) || comments.some(c => isHandoff(c) && afterBaseline(c))) {
        counters.filtered_handed_off++; reasons[number] = `handed off to maintainers after ${MAX_NUDGES_PER_PR} nudges`; continue;
      }

      const conflicting = isConflicting(pr);
      const headDate = new Date(pr.commits?.nodes?.[0]?.commit?.committedDate || 0).getTime();
      const nudges = comments.filter(c => isNudge(c) && afterBaseline(c));
      const lastNudgeAt = nudges.length ? new Date(nudges[0].created_at).getTime() : null;
      // Two different questions, decided in this order:
      //   nudgeCapped — has this PR already received the maximum number of nudges (this
      //                 cycle)? If so, it gets NO further nudge; the only remaining outcome
      //                 is the hand-off below (or "nothing actionable").
      //   Filter 4/5  — may ANOTHER nudge be posted right now?
      // Filter 4 must therefore not apply to a capped PR: a 6th nudge that Copilot never
      // answered (head unchanged) is precisely the state that has to reach the hand-off,
      // and the old `continue` here left such PRs stuck forever — no 7th nudge, no
      // hand-off, nothing visible. The cooldown (Filter 5) still applies: the hand-off is
      // evaluated on the same schedule a 7th nudge would have been, so the last nudge gets
      // the same 60 min to work as every earlier one.
      const nudgeCapped = nudges.length >= MAX_NUDGES_PER_PR;
      if (nudgeBaseline) core.info(`#${number}: ${HANDOFF_LABEL} was removed at ${new Date(nudgeBaseline).toISOString()}; counting ${nudges.length} nudge(s) since then`);
      // Filter 4 — upstream: never two actionable nudges in a row (except when conflicting).
      // tt-metal refinement: a nudge followed by a Copilot push that changed the head is a
      // NEW state (CI ran again on new code) and may be re-evaluated; only an unanswered
      // nudge (head unchanged since) blocks — and only one from THIS cycle: a nudge older
      // than the reset baseline belongs to the previous, handed-off cycle, and the
      // maintainer's label removal is itself the instruction to try again.
      const lastCommentIsUnansweredNudge = comments.length && isNudge(comments[0]) && afterBaseline(comments[0]);
      const noPushSinceLastNudge = lastCommentIsUnansweredNudge && headDate <= new Date(comments[0].created_at).getTime();
      // Second tt-metal refinement (PR #58341, 2026-09-29): a session that STARTS and FINISHES
      // (or fails) entirely after the nudge, with no push and no reply either, means Copilot
      // tried and silently produced nothing -- e.g. the "GitHub engine reported fetch failed"
      // write-back error observed there. `session`/`sessionAt` can only be a finished/failed
      // event here (a `copilot_work_started` with no later finished event already `continue`d
      // at Filter 2, unless stale). The old code treated this identically to "still working,
      // wait" and blocked forever: Filter 4 never lets a second nudge through, so a silently
      // failed nudge could never progress toward a working retry OR the nudge cap. A silent
      // session is evidence the last nudge already failed, not a reason to keep waiting on it.
      //
      // "No reply" here must mean no reply ANYWHERE, not just no top-level issue comment:
      // `comments` (from `issues.listComments`) does not include inline review-thread replies,
      // and Copilot answering a review thread without pushing is a real, non-silent outcome
      // (review round below already defines "Copilot replied" as the thread's last comment
      // being Copilot's -- mirrored here, not reinvented). `fetchThreadNodes()` is memoized so
      // this check and the later, unconditional review-thread section never fetch it twice.
      let silentSessionSinceNudge = false;
      if (noPushSinceLastNudge && sessionAt !== null &&
          (session === 'copilot_work_finished' || session === 'copilot_work_finished_failure') &&
          new Date(sessionAt).getTime() > new Date(comments[0].created_at).getTime()) {
        const nudgeTime = new Date(comments[0].created_at).getTime();
        const threadsForSilenceCheck = await fetchThreadNodes();
        const repliedInThread = threadsForSilenceCheck.some(t => {
          const last = t.lastComment?.nodes?.[0];
          const total = t.comments?.totalCount ?? 0;
          return total > 1 && isCopilotCodingAgent(last?.author?.login) && new Date(last?.createdAt || 0).getTime() > nudgeTime;
        });
        if (repliedInThread) {
          core.info(`#${number}: Copilot's session (${lastStartedAt || '?'} → ${sessionAt}, ${session}) replied in a review thread; not treating as silent`);
        } else {
          silentSessionSinceNudge = true;
          counters.silent_session_since_nudge = (counters.silent_session_since_nudge || 0) + 1;
          core.info(`#${number}: Copilot's session (${lastStartedAt || '?'} → ${sessionAt}, ${session}) after the last nudge produced no push and no reply anywhere; not blocking on it`);
        }
      }
      if (!nudgeCapped && noPushSinceLastNudge && !conflicting && !silentSessionSinceNudge) {
        counters.filtered_last_comment_from_sous_chef++; reasons[number] = 'last comment is an unanswered sous-chef nudge (no push since)'; continue;
      }
      // Filter 5 — cooldown since the last actionable nudge (also delays a hand-off).
      if (lastNudgeAt !== null && now - lastNudgeAt < COOLDOWN_MS) {
        counters.filtered_cooldown++;
        reasons[number] = nudgeCapped ? 'nudge cap reached; hand-off waits for the 60 min cooldown after the last nudge' : 'inside the 60 min cooldown';
        continue;
      }

      // Context for the agent.
      const failedForCopilot = [];
      const needsMaintainerApproval = [];
      for (const c of checks) {
        if (c.__typename === 'CheckRun') {
          if (c.status === 'WAITING' || c.conclusion === 'ACTION_REQUIRED') needsMaintainerApproval.push({ name: c.name, url: c.detailsUrl || null });
          else if (['FAILURE', 'TIMED_OUT', 'STARTUP_FAILURE'].includes(c.conclusion || '')) failedForCopilot.push({ name: c.name, conclusion: c.conclusion, url: c.detailsUrl || null });
        } else if (c.__typename === 'StatusContext' && ['FAILURE', 'ERROR'].includes(c.state || '')) {
          failedForCopilot.push({ name: c.context, conclusion: c.state, url: c.targetUrl || null });
        }
      }

      // ALL review threads. Reuses the Filter 4 silent-session check's fetch when it already
      // ran one (memoized in fetchThreadNodes above); otherwise this is the first fetch.
      const threadNodes = await fetchThreadNodes();

      // Commits of the PR (sha -> commit date) for `verifyFix`. Fetched only when a bot
      // thread could be resolvable, to spare API calls on simple PRs.
      let prCommits = null;
      const loadPrCommits = async () => {
        if (prCommits) return prCommits;
        const list = await github.paginate(github.rest.pulls.listCommits, { owner, repo, pull_number: number, per_page: 100 });
        prCommits = list.map(c => ({ sha: (c.sha || '').toLowerCase(), at: new Date(c.commit?.committer?.date || c.commit?.author?.date || 0).getTime() }));
        return prCommits;
      };
      // "Copilot replied" is decided by the thread's LATEST comment only: a reviewer
      // follow-up after Copilot's reply makes it unanswered again. Threads Copilot answered
      // without a verifiable fix stay open for the human reviewer.
      const threads = [];
      for (const t of threadNodes.filter(t => !t.isResolved)) {
        const first = t.firstComment?.nodes?.[0];
        const last = t.lastComment?.nodes?.[0];
        const total = t.comments?.totalCount ?? 0;
        const copilotReplied = total > 1 && isCopilotCodingAgent(last?.author?.login);
        const resolvableBot = isResolvableReviewerBot(first?.author);
        let fixVerified = false;
        if (resolvableBot && copilotReplied && !t.isOutdated) {
          const openedAt = new Date(first?.createdAt || 0).getTime();
          // Cheap pre-check before the commits call: any hex word at all?
          if (/\b[0-9a-f]{7,40}\b/i.test(last?.body || '')) fixVerified = verifyFix(last?.body, await loadPrCommits(), openedAt);
          if (!fixVerified) counters.bot_threads_answered_but_unverified++;
        }
        threads.push({
          id: t.id, path: t.path || null, url: first?.url || null, outdated: !!t.isOutdated,
          opened_at: first?.createdAt || null,
          reviewer: first?.author?.login || 'unknown', reviewer_is_resolvable_bot: resolvableBot,
          excerpt: excerpt(first?.body), comment_count: total, copilot_replied: copilotReplied,
          last_reply_by: total > 1 ? (last?.author?.login || 'unknown') : null, fix_verified: fixVerified
        });
      }
      // Decisions over the COMPLETE set first ...
      const resolvableBotThreads = threads.filter(t => t.fix_verified).map(t => t.id);
      const unansweredAll = threads.filter(t => !t.copilot_replied)
        .sort((a, b) => new Date(a.opened_at || 0) - new Date(b.opened_at || 0)); // oldest first
      // ... then truncation, for the prompt only.
      const unansweredForPrompt = unansweredAll.slice(0, MAX_THREADS_PER_PR).map(({ fix_verified, ...rest }) => rest);

      const zeroDiffStalled = (pr.changedFiles ?? -1) === 0 && now - new Date(pr.createdAt).getTime() >= ZERO_DIFF_AGE_MS;
      // A mention only starts a session on a PR that is still ASSIGNED to Copilot (GitHub
      // docs); a maintainer who unassigned Copilot has taken the PR over.
      const copilotAssigned = (pr.assignees?.nodes || []).some(a => isCopilotCodingAgent(a?.login));
      const wouldNeedNudge = conflicting || zeroDiffStalled || failedForCopilot.length > 0 || unansweredAll.length > 0;
      const needsNudge = copilotAssigned && wouldNeedNudge;

      // Filter 6 (new) — nothing actionable. A PR that is green and merely waiting for a
      // human reviewer (or only has runs stuck on the maintainer approval gate) has nothing
      // for Copilot to do and nothing for this workflow to resolve; the prompt's rule 4
      // would end in `noop`, so do not start a paid agent session for it. Same when Copilot
      // is no longer assigned: a nudge could not wake it.
      if (!needsNudge && resolvableBotThreads.length === 0) {
        if (wouldNeedNudge && !copilotAssigned) {
          counters.filtered_copilot_unassigned++;
          reasons[number] = 'Copilot is no longer assigned to this PR (a mention cannot start a session); left to its human owner';
        } else {
          counters.filtered_nothing_actionable++;
          reasons[number] = needsMaintainerApproval.length
            ? 'nothing for Copilot to do: only runs waiting on a maintainer to approve them'
            : 'nothing for Copilot to do: green and waiting on a human';
        }
        continue;
      }

      // Hand-off (new vs upstream), performed HERE, deterministically, not by the agent:
      // after MAX_NUDGES_PER_PR actionable nudges the PR still needs one. Post the
      // explanation first, then the label (comment-first for the same reason as
      // issue-monster.md's retry checkpoint: the state that hides the PR must never exist
      // without the explanation). Either write failing is retried next tick; the marker
      // check above prevents a duplicate comment.
      if (needsNudge && nudgeCapped) {
        if (handoffsThisRun >= MAX_ELIGIBLE) { reasons[number] = 'nudge cap reached; hand-off deferred to the next run (per-run cap)'; continue; }
        // These two writes go straight through REST, so gh-aw's `required-labels`
        // enforcement (which re-reads labels right before the add-comment / add-labels
        // handlers post) does not cover them. Apply the same rule by hand: a FRESH read of
        // the PR immediately before writing, not the `labels` snapshot from the top of this
        // iteration (several paginated calls ago, after other PRs' evaluations). If a
        // maintainer removed `copilot-flow` in the meantime, the kill switch wins and
        // nothing is posted; if another run already handed the PR off, or it was closed or
        // converted to a draft, there is nothing to do either.
        const fresh = (await github.rest.pulls.get({ owner, repo, pull_number: number })).data;
        const freshLabels = (fresh?.labels || []).map(l => (l?.name || '').toLowerCase());
        const staleReason = !freshLabels.includes(FLOW_LABEL) ? `${FLOW_LABEL} was removed before the hand-off could be posted (kill switch honoured)`
          : freshLabels.includes(HANDOFF_LABEL) ? 'already handed off by another run'
          : fresh?.state !== 'open' || fresh?.draft ? 'no longer open and non-draft at write time'
          : null;
        if (staleReason) {
          counters.handoff_skipped_stale_state++; reasons[number] = `nudge cap reached; hand-off skipped: ${staleReason}`;
          core.info(`#${number}: hand-off skipped: ${staleReason}`);
          continue;
        }
        const body = buildHandoffComment({
          pr, nudgeCount: nudges.length, conflicting, zeroDiffStalled, failedForCopilot, unansweredAll, needsMaintainerApproval, runUrl, now,
          silentSession: silentSessionSinceNudge ? { startedAt: lastStartedAt, endedAt: sessionAt, outcome: session } : null
        });
        try {
          await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
          core.info(`#${number}: posted the hand-off comment (${nudges.length} nudges)`);
          try {
            await github.rest.issues.addLabels({ owner, repo, issue_number: number, labels: [HANDOFF_LABEL] });
          } catch (labelError) {
            // Label is best effort (it may not exist yet); the marker comment alone is
            // sufficient for Filter 3. But the label is also the documented RESET (its
            // removal is the nudge-count baseline), so without it the PR can only be put
            // back by deleting the comment, and its old nudges would then still count.
            core.warning(`#${number}: hand-off comment posted but label ${HANDOFF_LABEL} could not be applied: ${labelError.message}. Create the label: without it maintainers cannot reset the nudge count for this PR.`);
          }
          counters.handed_off_now++; handoffsThisRun++;
          reasons[number] = `handed off to maintainers after ${nudges.length} nudges (this run)`;
        } catch (error) {
          core.warning(`#${number}: could not post the hand-off comment: ${error.message}; retried next tick`);
          reasons[number] = 'nudge cap reached; hand-off comment failed, retried next run';
        }
        continue;
      }

      // Priority: 0 conflict, 1 zero-diff stall, 2 needs a nudge (failed checks or threads
      // whose latest comment is not Copilot's), 3 only bot threads to resolve.
      const bucket = !needsNudge ? 3 : conflicting ? 0 : zeroDiffStalled ? 1 : 2;

      eligible.push({
        number, title: pr.title, url: pr.url, headRefOid: pr.headRefOid, headRefName: pr.headRefName,
        createdAt: pr.createdAt, updatedAt: pr.updatedAt, changedFiles: pr.changedFiles,
        mergeStateStatus: pr.mergeStateStatus, mergeable: pr.mergeable, merge_conflict: conflicting,
        reviewDecision: pr.reviewDecision, copilot_assigned: copilotAssigned,
        copilot_session: session, zero_diff_stalled: zeroDiffStalled,
        nudge_count: nudges.length, needs_nudge: needsNudge,
        failed_checks_for_copilot: failedForCopilot, checks_needing_maintainer_approval: needsMaintainerApproval,
        unanswered_thread_count: unansweredAll.length,
        unanswered_threads_not_shown: unansweredAll.length - unansweredForPrompt.length,
        unanswered_threads: unansweredForPrompt, resolvable_bot_threads: resolvableBotThreads,
        priority_bucket: bucket
      });
    } catch (error) {
      counters.filtered_error++; reasons[number] = `could not evaluate: ${error.message}`;
      core.warning(`#${number}: ${error.message}`);
    }
  }

  // Deterministic priority (applied here instead of by the agent): conflicts, zero-diff
  // stalls, PRs that need a nudge, PRs with only bot threads to resolve, then most recently
  // updated; lower PR number breaks ties. (Upstream ranks "human threads the author
  // answered" third because it resolves those; this port never resolves human threads, so
  // an answered thread is not actionable here.)
  eligible.sort((a, b) => a.priority_bucket - b.priority_bucket
    || new Date(b.updatedAt) - new Date(a.updatedAt) || a.number - b.number);
  const selected = eligible.slice(0, MAX_ELIGIBLE);
  counters.eligible = selected.length;

  // Slash-command context: what happened to the PR the command was posted on.
  let trigger = null;
  if (context.eventName === 'issue_comment' && context.payload?.issue?.number) {
    const n = context.payload.issue.number;
    const status = selected.some(p => p.number === n) ? 'eligible'
      : eligible.some(p => p.number === n) ? `eligible but beyond the ${MAX_ELIGIBLE}-PR per-run cap`
      : reasons[n] || `not an open, non-draft, ${FLOW_LABEL}-labeled Copilot PR`;
    trigger = { number: n, status };
    // The pinned resolve-pull-request-review-thread handler scopes resolution to the
    // TRIGGERING PR whenever the run has one (issue_comment); threads on the other selected
    // PRs would be rejected. Defer them to the next scheduled tick.
    for (const p of selected) if (p.number !== n && p.resolvable_bot_threads.length) {
      counters.resolvable_threads_deferred += p.resolvable_bot_threads.length;
      p.resolvable_bot_threads = [];
    }
  }

  // Keep the total number of thread IDs the prompt asks the agent to resolve within the
  // safe-output handler's budget (RESOLVE_THREADS_MAX == resolve-pull-request-review-
  // thread.max). Higher-priority PRs keep theirs; the rest are deferred.
  let resolveBudget = RESOLVE_THREADS_MAX;
  for (const p of selected) {
    const keep = p.resolvable_bot_threads.slice(0, Math.max(0, resolveBudget));
    counters.resolvable_threads_deferred += p.resolvable_bot_threads.length - keep.length;
    p.resolvable_bot_threads = keep;
    resolveBudget -= keep.length;
  }
  if (counters.resolvable_threads_deferred) core.info(`${counters.resolvable_threads_deferred} resolvable bot thread(s) deferred to the next run (cap ${RESOLVE_THREADS_MAX} / trigger scoping)`);

  core.setOutput('eligible_count', String(selected.length));
  core.setOutput('eligible_numbers', selected.map(p => p.number).join(','));
  core.setOutput('eligible_context', JSON.stringify(selected));
  core.setOutput('trigger_context', trigger ? JSON.stringify(trigger) : '');
  core.setOutput('counters', JSON.stringify(counters));
  const rows = Object.entries(counters).map(([k, v]) => `| ${k} | ${v} |`).join('\n');
  await core.summary.addHeading('PR Sous Chef — prefilter', 3)
    .addRaw(`\nNudges recognised as authored by \`${identity.login}\`.\n\n| Metric | Count |\n|---|---|\n${rows}\n\nSelected: ${selected.map(p => `#${p.number} (bucket ${p.priority_bucket})`).join(', ') || 'none'}\n`)
    .write();
  return { selected, counters, reasons };
}

// Deterministic hand-off comment (no at-mention anywhere: it must not start a session, and
// must not look like a nudge to `isNudge`, which is why "Copilot nudges" is written without
// the `@`).
function buildHandoffComment({ pr, nudgeCount, conflicting, zeroDiffStalled, failedForCopilot, unansweredAll, needsMaintainerApproval, runUrl, now, silentSession = null }) {
  const hours = Math.round((now - new Date(pr.createdAt).getTime()) / 3600000);
  const lines = [
    HANDOFF_MARKER,
    `🍳 **PR Sous Chef — handing off to maintainers.** This PR has received ${nudgeCount} automated Copilot nudges (cap ${MAX_NUDGES_PER_PR}) without becoming mergeable, so PR Sous Chef will not nudge it again.`,
    '',
    'Still outstanding:'
  ];
  if (conflicting) lines.push('- Merge conflict with `main` (`mergeStateStatus: DIRTY`).');
  if (zeroDiffStalled) lines.push(`- No files changed ${hours} hours after the PR was opened.`);
  for (const c of failedForCopilot.slice(0, 10)) lines.push(`- Failing check: ${c.url ? `[${c.name}](${c.url})` : c.name} (${c.conclusion}).`);
  if (failedForCopilot.length > 10) lines.push(`- … and ${failedForCopilot.length - 10} more failing check(s).`);
  for (const t of unansweredAll.slice(0, 10)) lines.push(`- Review thread by ${t.reviewer} with no reply from Copilot${t.url ? `: ${t.url}` : ''}`);
  if (unansweredAll.length > 10) lines.push(`- … and ${unansweredAll.length - 10} more unanswered review thread(s).`);
  for (const c of needsMaintainerApproval.slice(0, 5)) lines.push(`- Waiting for a maintainer to approve the run: ${c.url ? `[${c.name}](${c.url})` : c.name}.`);
  // Distinguishes "Copilot never got the nudge" from "Copilot tried and its write-back silently
  // failed" -- without this a maintainer has to reconstruct the timeline by hand (as happened
  // on PR #58341) to tell the two apart.
  if (silentSession) {
    lines.push(`- The last nudge's Copilot session ran (${silentSession.startedAt || '?'} → ${silentSession.endedAt || '?'}, ended \`${silentSession.outcome}\`) but produced no new commit and no reply -- its write-back likely failed rather than nothing happening.`);
  }
  lines.push('',
    `A maintainer should take the PR over, give Copilot direct guidance in a comment (an at-mention from a maintainer with write access starts a new session), or close it. To put the PR back under PR Sous Chef, remove the \`${HANDOFF_LABEL}\` label: only nudges posted after that removal count toward the next cap of ${MAX_NUDGES_PER_PR}, so this comment can stay as history (deleting it is neither needed nor sufficient).`,
    '',
    `> 🍳 *Posted by the [PR Sous Chef](${runUrl}) pre-activation check — automated; no agent was involved in this decision.*`);
  return lines.join('\n');
}

module.exports = {
  run,
  // exported for prefilter.test.js
  FLOW_LABEL, HANDOFF_LABEL, HANDOFF_MARKER, WORKFLOW_ID, MAX_NUDGES_PER_PR, MAX_ELIGIBLE,
  MAX_THREADS_PER_PR, RESOLVE_THREADS_MAX, REPLY_VETO, COOLDOWN_MS, SESSION_STALE_MS, PRE_ACTIVATION_LOGIN,
  matchesWorkflowId, isCopilotCodingAgent, isResolvableReviewerBot, isConflicting,
  makeIdentity, verifyFix, buildHandoffComment
};
