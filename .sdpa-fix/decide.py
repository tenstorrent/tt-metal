#!/usr/bin/env python3
"""Slack decision buttons for the SDPA autofix bot (Socket Mode listener).

When a fix is a human call, fixer.sh opens nothing and posts a message with a
button per option plus Reject. This process holds an OUTBOUND websocket to
Slack (Socket Mode: no public URL needed), and on a click:

  1. checks the clicker is allowed to decide (DECIDERS; empty = anyone in the
     private channel),
  2. checks the signature is still awaiting a decision,
  3. records {key, by, at} as decision_choice in the ledger,
  4. edits the message: buttons replaced by "decided: … by @user, applying…",
  5. starts `DECISIONS_ONLY=1 fixer.sh`, which applies the choice now.

"✍️ Other…" opens a modal; the submitted text is stored as decision_note
(state "revise"), and the fixer has the agent build a NEW poll from it, which
replaces the buttons in the same message. Any number of rounds.

Run by decide.sh (cron every 5 min, flock-guarded), which supplies the env:
SLACK_APP_TOKEN (xapp-, connections:write), SLACK_BOT_TOKEN, DECIDERS.
"""
import datetime
import json
import os
import subprocess
import sys
import time

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

HOME = os.path.expanduser("~/.sdpa-fix")
FIXLIB = [sys.executable, os.path.join(HOME, "fixlib.py")]
DECIDERS = set(os.environ.get("DECIDERS", "").split())


def log(msg):
    print("[%s] %s" % (datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), msg), flush=True)


def ledger_record(sig):
    out = subprocess.run(FIXLIB + ["get", sig], capture_output=True, text=True, check=False)
    try:
        return json.loads(out.stdout).get(sig) or {}
    except ValueError:
        return {}


def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def run_fixer():
    env = dict(os.environ, DECISIONS_ONLY="1")
    with open(os.path.join(HOME, "logs", "decide-runs.log"), "a") as out:
        subprocess.Popen(["bash", os.path.join(HOME, "fixer.sh")], env=env, stdout=out, stderr=out,
                         start_new_session=True)


def open_note_modal(web, p, sig, channel, ts):
    web.views_open(trigger_id=p["trigger_id"], view={
        "type": "modal", "callback_id": "autofix_note",
        "private_metadata": json.dumps({"sig": sig, "channel": channel, "ts": ts}),
        "title": {"type": "plain_text", "text": "Autofix: other option"},
        "submit": {"type": "plain_text", "text": "Send to agent"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [{"type": "input", "block_id": "note",
                    "label": {"type": "plain_text", "text": "What should the agent do instead?"},
                    "hint": {"type": "plain_text",
                             "text": "The agent turns this into a new set of options. Nothing is opened until you pick one."},
                    "element": {"type": "plain_text_input", "action_id": "text", "multiline": True,
                                "max_length": 1500}}]})


def handle_note(client, p):
    web = client.web_client
    user = p["user"]["id"]
    meta = json.loads(p["view"].get("private_metadata") or "{}")
    sig, channel, ts = meta.get("sig", ""), meta.get("channel", ""), meta.get("ts", "")
    text = ((p["view"]["state"]["values"].get("note") or {}).get("text") or {}).get("value") or ""
    text = text.strip()
    if not text or (DECIDERS and user not in DECIDERS):
        return
    rec = ledger_record(sig)
    if rec.get("state") != "awaiting_decision" or rec.get("decision_choice"):
        web.chat_postEphemeral(channel=channel, user=user, text="This question was already answered or is no longer open.")
        return
    subprocess.run(FIXLIB + ["mark", "--state", "revise", "--extra",
                             json.dumps({"decision_note": {"text": text, "by": user, "at": now()}}), sig], check=True)
    log("note on %s by %s: %s" % (sig, user, text[:120]))
    try:
        web.chat_update(channel=channel, ts=ts, text="preparing new options",
                        blocks=[{"type": "section", "text": {"type": "mrkdwn",
                                 "text": "❓ *autofix — needs your decision*\n✍️ <@%s>: “%s”\n_preparing new options…_"
                                         % (user, text[:500])}}])
    except Exception as e:
        log("WARN: message edit failed: %s" % e)
    run_fixer()


def handle(client, req):
    # Ack first: Slack retries anything not acked within 3 s (for a modal
    # submission an empty ack also closes the modal).
    client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
    if req.type != "interactive":
        return
    if req.payload.get("type") == "view_submission" and \
            req.payload.get("view", {}).get("callback_id") == "autofix_note":
        handle_note(client, req.payload)
        return
    if req.payload.get("type") != "block_actions":
        return
    p = req.payload
    act = (p.get("actions") or [{}])[0]
    if not act.get("action_id", "").startswith("autofix_decide_"):
        return
    try:
        v = json.loads(act["value"])
    except (KeyError, ValueError):
        return
    user = p["user"]["id"]
    channel = p["channel"]["id"]
    msg = p.get("message") or {}
    web = client.web_client
    sig, key = v.get("sig", ""), v.get("key", "")

    if DECIDERS and user not in DECIDERS:
        web.chat_postEphemeral(channel=channel, user=user,
                               text="Only the configured deciders can answer autofix questions.")
        log("refused click by %s on %s/%s (not a decider)" % (user, sig, key))
        return
    rec = ledger_record(sig)
    if rec.get("state") != "awaiting_decision" or rec.get("decision_choice"):
        web.chat_postEphemeral(channel=channel, user=user,
                               text="This question was already answered or is no longer open.")
        log("ignored click by %s on %s/%s (state %s)" % (user, sig, key, rec.get("state")))
        return
    if key == "OTHER":
        open_note_modal(web, p, sig, channel, msg.get("ts"))
        return

    choice = {"key": key, "by": user, "at": now()}
    subprocess.run(FIXLIB + ["mark", "--state", "keep", "--extra", json.dumps({"decision_choice": choice}), sig],
                   check=True)
    label = "Reject" if key == "REJECT" else next(
        ("%s · %s" % (o["key"], o["label"]) for o in rec.get("decision", {}).get("options", []) if o["key"] == key), key)
    log("decision %s on %s by %s" % (key, sig, user))

    # Keep the question, replace the buttons with who decided what.
    blocks = [b for b in (msg.get("blocks") or []) if b.get("type") != "actions"]
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
                   "text": "✅ decided: *%s* by <@%s>, applying…" % (label, user)}]})
    try:
        web.chat_update(channel=channel, ts=msg.get("ts"), text=msg.get("text", "decided"), blocks=blocks)
    except Exception as e:  # the ledger is the source of truth; a failed edit is cosmetic
        log("WARN: message edit failed: %s" % e)

    run_fixer()


def main():
    app, bot = os.environ.get("SLACK_APP_TOKEN", ""), os.environ.get("SLACK_BOT_TOKEN", "")
    if not app.startswith("xapp-") or not bot:
        log("FATAL: SLACK_APP_TOKEN (xapp-) and SLACK_BOT_TOKEN are required")
        sys.exit(1)
    client = SocketModeClient(app_token=app, web_client=WebClient(token=bot))
    client.socket_mode_request_listeners.append(handle)
    client.connect()
    log("listening for decision buttons (deciders: %s)" % (" ".join(sorted(DECIDERS)) or "anyone in channel"))
    while True:
        time.sleep(300)


if __name__ == "__main__":
    main()
