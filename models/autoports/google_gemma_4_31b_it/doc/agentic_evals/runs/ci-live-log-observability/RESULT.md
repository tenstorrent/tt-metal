# CI live-log observability: no bounded-latency viewer proved

The existing personal gh authorization can download partial job logs, but the observed download channel is too stale to establish a reliable no-score dry window for short singleton SWE tasks. No alternative authenticated live viewer was demonstrated with the credentials/tools already available. This refutes treating the singleton route as **operationally ready**; it does not prove GitHub has no live browser viewer.

## Direct observations

Authorized active run42234274300/job127947875918 was inspected read-only. `probe.json` stores only timestamps/status/byte counts:

| Observation | Newest log event | Event age when fetched |
|---|---|---:|
| Standard REST download,16:09:11.005UTC |16:07:55.712UTC|75.293s|
| REST with Range and Cache-Control:no-cache,16:10:35.518UTC|16:07:55.712UTC|159.806s|

The second request still returned the same roughly33.7MB snapshot, HTTP200, despite the requested byte range/no-cache. These headers did not force a new server-side upload or expose a streaming tail. Its event age already exceeds the prior Requests task's entire151.802s agent runtime. The coordinator's earlier observation—last-modified16:02:43 at fetch16:06:53—is additional evidence of variable batching, not a measured upper bound. Neither75s nor160s is an SLA; faster polling cannot reveal lines that have not reached the downloaded blob.

`gh api` against the private HTML job page, using the same existing gh authentication, returned404 with no logged-in marker/live-log element. The ordinary REST job query succeeds. Thus the working API authentication did not establish an authenticated browser session for this page. No login, cookie extraction, token refresh, account fallback or new credential was attempted. Installed gh is2.4.0; its missing `gh auth token` subcommand was encountered harmlessly, and subsequent calls used gh's own credential handling without printing secrets.

## Official mechanisms inspected

The [workflow-job REST documentation](https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run) documents a redirect to a text download with an expiring link. It does not document a bounded-freshness live subscription. The [workflow logs documentation](https://docs.github.com/en/actions/how-tos/monitor-workflows/use-workflow-run-logs) describes logged-in web access and downloads, not converting a personal API token into a browser session.

Official runner source at `cf65f3ec8100ee0046d735ac5d647e00f4b8e817` does contain a faster console publishing path:

- [JobServerQueue.cs](https://github.com/actions/runner/blob/cf65f3ec8100ee0046d735ac5d647e00f4b8e817/src/Runner.Common/JobServerQueue.cs):32-36 uses250/500ms console dequeue delays;:414-418 sends console batches to ResultsServer/JobServer feed APIs.
- [ResultsServer.cs](https://github.com/actions/runner/blob/cf65f3ec8100ee0046d735ac5d647e00f4b8e817/src/Runner.Common/ResultsServer.cs):56-64 receives a job live-console URL/token, :228-284 sends websocket data, and:287-294 attaches the runner-provided bearer token. This is **runner-to-service publishing**, not a repository reader subscription. Its flush cadence is not end-to-end observer latency. Runner job tokens/URLs are neither needed nor authorized to extract for this investigation, and were not used.
- Official [gh run view source](https://github.com/cli/cli/blob/eeb44bc432c0107de002dadd34a61b4e2bd15882/pkg/cmd/run/view/view.go):313-320 explicitly rejects `--log` while a selected job/run is incomplete. `gh api` partial downloads therefore expose more ongoing data than that command, but not a streaming viewer.

An unauthenticated **public** official actions/runner job page was inspected to understand the page mechanism. Its workflow logs links require sign-in. GitHub-served asset `workflow-run-2a5a1a483edf22ee.js` uses same-origin credentialed JSON fetch, obtains an `authenticated_url`, then `logStreamWebSocketUrl` for run-step progress; newer results updates use authenticated socket events. Those browser-side names show a separate web authorization mechanism, not a usable public REST log endpoint or a demonstrated log-content stream with our API credentials. No authenticated URL or websocket credential was obtained, stored in this result or followed. Source snapshots remain private under `bringup/artifacts/ci-live-log-observability-source/`; exact repository pins are in `source-pins.json`.

## Consequence for the proposed dry route

There is currently no verified read-only channel with latency bounded below a short task's remaining execution window. A snapshot showing the second singleton request can arrive after that task has already finished and been graded. Checking `job.status=in_progress` or the long outer CI “Run tests” step does not exclude that: setup, other work and cleanup can keep the job running after the agent finishes. Absence of a final score in a stale blob is not proof that no score exists.

Do not dispatch a frozen singleton dry assuming this channel will permit an early no-score stop, and do not change task limits or insert sleeps to create an observation window. This finding leaves the existing route question unresolved. A separately authorized, already-authenticated browser viewer with measured freshness could change the result, but no such available session/tool was found or created here. Alternatively the operator could explicitly define another acceptable dry procedure; that exception remains outside this investigation. Long GPQA observation windows remain a different case and are not invalidated by the short-task counterexample.

Device use0. No dispatch, cancellation, runner/workflow/harness modification or central evidence update occurred. Tokens, signed log URLs, raw logs and credentials are excluded from this compact report and probe metadata.
