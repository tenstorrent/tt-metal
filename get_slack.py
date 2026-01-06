import json
import os
import time
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# --- CONFIGURATION ---
SLACK_TOKEN = ""
CHANNEL_ID = ""
START_DATE_STR = "December 17, 2025"
OUTPUT_FILE = "build_slack_export_with_threads.json"


def get_unix_timestamp(date_string):
    dt = datetime.strptime(date_string, "%B %d, %Y")
    return time.mktime(dt.timetuple())


def fetch_replies(client, channel_id, thread_ts):
    """Fetches all replies for a specific thread."""
    replies = []
    cursor = None
    while True:
        try:
            response = client.conversations_replies(channel=channel_id, ts=thread_ts, cursor=cursor, limit=1000)
            time.sleep(0.5)
            # The first message in the replies list is actually the parent message.
            # We filter it out to avoid duplication within the 'replies' list.
            batch = [msg for msg in response["messages"] if msg.get("ts") != thread_ts]
            replies.extend(batch)

            if response["has_more"]:
                cursor = response["response_metadata"]["next_cursor"]
            else:
                break
        except SlackApiError as e:
            print(f"Error fetching replies for thread {thread_ts}: {e}")
            break
    return replies


def download_messages():
    # Delete output file if it already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Deleting existing {OUTPUT_FILE}...")
        os.remove(OUTPUT_FILE)

    client = WebClient(token=SLACK_TOKEN)
    oldest_timestamp = get_unix_timestamp(START_DATE_STR)

    all_messages = []
    cursor = None

    print(f"Fetching messages from {START_DATE_STR}...")

    # First pass: collect all messages without fetching replies
    while True:
        try:
            response = client.conversations_history(
                channel=CHANNEL_ID, oldest=str(oldest_timestamp), cursor=cursor, limit=1000
            )

            messages = response["messages"]
            all_messages.extend(messages)

            print(f"Retrieved {len(messages)} parent messages... (Total so far: {len(all_messages)})")

            if response["has_more"]:
                cursor = response["response_metadata"]["next_cursor"]
            else:
                break

        except SlackApiError as e:
            # Handle rate limiting (Slack tier 3/4 limits)
            if e.response.status_code == 429:
                delay = int(e.response.headers.get("Retry-After", 10))
                print(f"Rate limited. Sleeping for {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Error fetching conversations: {e}")
                break

    # Count messages with replies
    messages_with_replies = [msg for msg in all_messages if "thread_ts" in msg and msg.get("reply_count", 0) > 0]
    total_with_replies = len(messages_with_replies)

    print(f"\nFound {total_with_replies} messages with replies. Fetching replies...")

    # Second pass: fetch replies with progress tracking
    current_index = 0
    for msg in all_messages:
        # Check if this message is the start of a thread
        # 'reply_count' indicates there are replies to fetch
        if "thread_ts" in msg and msg.get("reply_count", 0) > 0:
            current_index += 1
            print(f"  --> Fetching replies from notification {current_index}/{total_with_replies} (ts: {msg['ts']})...")
            msg["replies"] = fetch_replies(client, CHANNEL_ID, msg["thread_ts"])

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=4, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_messages)} parent messages (with nested replies) to {OUTPUT_FILE}")


if __name__ == "__main__":
    download_messages()
