# Read Latest Slack Message (Test)

## Overview
Print the newest message from Slack public channel `C08SJ7MGESY` using Slack Web API.

## Prerequisites
- `SLACK_BOT_TOKEN` is set in the shell environment.
- Bot token has `channels:history`.
- Bot is a member of channel `C08SJ7MGESY`.
- `jq` is available.

## Steps
1. Follow `.cursor/rules/ci-slack-read-latest-message.mdc`.
2. Validate token is present:
   - `test -n "$SLACK_BOT_TOKEN"`
3. Fetch latest message:
   - `curl -sS -H "Authorization: Bearer $SLACK_BOT_TOKEN" "https://slack.com/api/conversations.history?channel=C08SJ7MGESY&limit=1"`
4. If `ok` is false, print `.error` and stop.
5. Print the latest message fields:
   - `ts`
   - `user` (or `bot_id`)
   - `text`
6. Optionally print permalink for the message:
   - `chat.getPermalink` with the same channel and message `ts`.

## Canonical Command Snippet
```bash
resp="$(curl -sS -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  "https://slack.com/api/conversations.history?channel=C08SJ7MGESY&limit=1")"

ok="$(printf '%s' "$resp" | jq -r '.ok')"
if [ "$ok" != "true" ]; then
  printf 'Slack API error: %s\n' "$(printf '%s' "$resp" | jq -r '.error')"
  exit 1
fi

ts="$(printf '%s' "$resp" | jq -r '.messages[0].ts')"
author="$(printf '%s' "$resp" | jq -r '.messages[0].user // .messages[0].bot_id // "unknown"')"
text="$(printf '%s' "$resp" | jq -r '.messages[0].text // ""')"

printf 'channel: C08SJ7MGESY\nts: %s\nauthor: %s\ntext: %s\n' "$ts" "$author" "$text"

plink_resp="$(curl -sS -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  --get "https://slack.com/api/chat.getPermalink" \
  --data-urlencode "channel=C08SJ7MGESY" \
  --data-urlencode "message_ts=$ts")"
printf 'permalink: %s\n' "$(printf '%s' "$plink_resp" | jq -r '.permalink // ""')"
```
