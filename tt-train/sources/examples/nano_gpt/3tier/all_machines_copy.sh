#!/bin/bash

# ====== Configuration ======
USER="ttuser"
SESSION_NAME="docker"
CONTAINER_NAME="$USER-host-mapped"
IMAGE="ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest"
BUILD_DIR="/home/$USER/git/tt-metal"
DEST_DIR="/home/$USER/git/"

MACHINES=(
    metal-wh-03
    metal-wh-04
    metal-wh-05
    metal-wh-06
)

DOCKER_CMD="sudo docker run -it \
  --name $CONTAINER_NAME \
  --pid=host \
  --network=host \
  --rm \
  -v /home/$USER/.ssh:/root/.ssh \
  -v /home/$USER:/home/$USER \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent \
  $IMAGE bash"

# ====== Mode Flags ======
RUN_MODE=false
KILL_MODE=false
RESTART_HARD=false
RESTART_SOFT=false
SYNC_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run) RUN_MODE=true ;;
        --kill) KILL_MODE=true ;;
        --restart) RESTART_HARD=true ;;
        --restart-soft) RESTART_SOFT=true ;;
        --sync) SYNC_MODE=true ;;
        *) echo "❌ Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ====== Check if HOST is Local ======
is_local_host() {
    local TARGET="$1"
    local LOCAL_HOSTNAMES=(
        "$(hostname)"
        "$(hostname -f)"
        "localhost"
    )
    local LOCAL_IPS=($(hostname -I))

    for item in "${LOCAL_HOSTNAMES[@]}" "${LOCAL_IPS[@]}"; do
        if [[ "$item" == "$TARGET" ]]; then
            return 0  # is local
        fi
    done
    return 1  # is remote
}

# ====== Rsync Build Directory ======
sync_build() {
    local HOST=$1

    if is_local_host "$HOST"; then
        echo "⚠️ [$HOST] Skipping sync (local machine)"
        return
    fi

    echo "📦 [$HOST] Syncing build directory..."
    local CMD="rsync -az --delete -e ssh \"$BUILD_DIR\" \"${USER}@${HOST}:${DEST_DIR}\""
    if [[ $RUN_MODE == true ]]; then
        eval $CMD && echo "✅ [$HOST] Sync complete" || echo "❌ [$HOST] Sync failed"
    else
        echo "[DRY-RUN] $CMD"
    fi
}

# ====== Docker & Tmux Control ======
remote_action() {
    local HOST=$1
    local FULL_SESSION="${SESSION_NAME}_${HOST//./_}"
    local SSH_CMD="ssh ${USER}@${HOST}"

    if [[ $KILL_MODE == true ]]; then
        if [[ $RUN_MODE == true ]]; then
            echo "🛑 [$HOST] Killing tmux and Docker..."
        $SSH_CMD "tmux kill-session -t $FULL_SESSION 2>/dev/null; sudo docker rm -f $CONTAINER_NAME 2>/dev/null" && \
            echo "✅ [$HOST] Killed" || echo "⚠️ [$HOST] Nothing to kill or failed"
            return
        else
            echo "[DRY-RUN] ssh $USER@$HOST 'tmux kill-session -t $FULL_SESSION; sudo docker rm -f $CONTAINER_NAME'"
            return
        fi
    fi

    if [[ $RESTART_HARD == true ]]; then
        if [[ $RUN_MODE == true ]]; then
            echo "🔄 [$HOST] Restarting (with pull)..."
            $SSH_CMD "
                tmux kill-session -t $FULL_SESSION 2>/dev/null;
                sudo docker rm -f $CONTAINER_NAME 2>/dev/null;
                sudo docker pull $IMAGE;
                tmux new-session -d -s $FULL_SESSION \"$DOCKER_CMD\";

            " && echo "✅ [$HOST] Restarted" || echo "❌ [$HOST] Restart failed"
            return
          else
            echo "[DRY-RUN] ssh $USER@$HOST 'tmux kill-session -t $FULL_SESSION; sudo docker rm -f $CONTAINER_NAME; sudo docker pull $IMAGE; tmux new-session -d -s $FULL_SESSION \"$DOCKER_CMD\"'"
            return
        fi
    fi

    if [[ $RESTART_SOFT == true ]]; then
        if [[ $RUN_MODE == true ]]; then
            echo "🔁 [$HOST] Restarting (no pull)..."
            $SSH_CMD "
                tmux kill-session -t $FULL_SESSION 2>/dev/null;
                sudo docker rm -f $CONTAINER_NAME 2>/dev/null;
                tmux new-session -d -s $FULL_SESSION \"$DOCKER_CMD\"
            " && echo "✅ [$HOST] Soft restarted" || echo "❌ [$HOST] Restart failed"
            return
          else
            echo "[DRY-RUN] ssh $USER@$HOST 'tmux kill-session -t $FULL_SESSION; sudo docker rm -f $CONTAINER_NAME; tmux new-session -d -s $FULL_SESSION \"$DOCKER_CMD\"'"
            return
        fi
    fi

}

# ====== Dispatcher ======
for HOST in "${MACHINES[@]}"; do
    (
        [[ $SYNC_MODE == true ]] && sync_build "$HOST"
        if [[ $KILL_MODE == true || $RESTART_HARD == true || $RESTART_SOFT == true ]]; then
            remote_action "$HOST"
        fi
    ) &
done

wait
echo "🎯 All hosts processed."
