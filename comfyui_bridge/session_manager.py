# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Thread-safe session management for per-step denoising workflows.

Manages session lifecycle across multiple per-step API calls, including:
- Session creation with model identification
- Activity tracking and timeout cleanup
- Thread-safe access with RLock
- Automatic expiration of idle sessions
- Session statistics and metadata

The SessionManager handles the stateless bridge pattern where ComfyUI controls
the workflow step-by-step, and sessions maintain just enough state to coordinate
across calls.

Key Classes:
    DenoiseSession: Dataclass representing a single denoising session
    SessionManager: Thread-safe manager for session lifecycle

Usage Example:
    >>> manager = SessionManager(timeout_seconds=1800)
    >>> session_id = manager.create_session(model_id="sdxl", total_steps=20)
    >>> session = manager.get_session(session_id)
    >>> manager.update_activity(session_id)
    >>> stats = manager.complete_session(session_id)
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import time

logger = logging.getLogger(__name__)


@dataclass
class DenoiseSession:
    """Represents a single denoising session across multiple steps."""

    session_id: str
    model_id: str
    created_at: datetime
    last_activity: datetime
    current_step: int = 0
    total_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate session on creation."""
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")


class SessionManager:
    """
    Thread-safe manager for denoising sessions.

    Handles:
    - Session creation and lifecycle management
    - Activity tracking with automatic timeout cleanup
    - Thread-safe access with RLock
    - Cleanup of expired sessions via background thread

    The manager uses UUIDs for session identification and tracks activity
    timestamps to clean up stale sessions automatically.
    """

    def __init__(self, timeout_seconds: int = 1800, cleanup_interval_seconds: int = 60):
        """
        Initialize the SessionManager.

        Args:
            timeout_seconds: Session expiration time in seconds (default 30 minutes)
            cleanup_interval_seconds: How often to run cleanup (default 60 seconds)
        """
        self.sessions: Dict[str, DenoiseSession] = {}
        self.timeout_seconds = timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._lock = threading.RLock()

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup, daemon=True, name="SessionCleanupThread"
        )
        self._cleanup_thread.start()
        logger.info(
            f"SessionManager initialized with timeout={timeout_seconds}s, "
            f"cleanup_interval={cleanup_interval_seconds}s"
        )

    def create_session(self, model_id: str, total_steps: int, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new denoising session.

        Args:
            model_id: Model identifier (e.g., "sdxl", "sd1.5", "sd3.5")
            total_steps: Total number of denoising steps expected
            metadata: Optional metadata dict (e.g., seed, scheduler info)

        Returns:
            Session ID (UUID string)

        Raises:
            ValueError: If model_id or total_steps invalid
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = DenoiseSession(
            session_id=session_id,
            model_id=model_id,
            created_at=now,
            last_activity=now,
            total_steps=total_steps,
            current_step=0,
            metadata=metadata or {},
        )

        with self._lock:
            self.sessions[session_id] = session

        logger.info(f"Created session {session_id} for model {model_id} " f"with {total_steps} steps")
        return session_id

    def get_session(self, session_id: str) -> Optional[DenoiseSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: UUID of the session

        Returns:
            DenoiseSession if found, None otherwise
        """
        with self._lock:
            return self.sessions.get(session_id)

    def is_session_valid(self, session_id: str) -> bool:
        """
        Check if a session exists and is not expired.

        Args:
            session_id: UUID of the session

        Returns:
            True if session exists and is valid, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False

        time_since_activity = datetime.now() - session.last_activity
        is_valid = time_since_activity.total_seconds() < self.timeout_seconds

        if not is_valid:
            logger.warning(f"Session {session_id} expired " f"({time_since_activity.total_seconds():.1f}s idle)")

        return is_valid

    def update_activity(self, session_id: str) -> bool:
        """
        Update the last activity timestamp for a session.

        Call this at the start of each per-step API call to prevent
        timeout-based cleanup.

        Args:
            session_id: UUID of the session

        Returns:
            True if updated, False if session not found or expired
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot update activity: session {session_id} not found")
            return False

        with self._lock:
            session.last_activity = datetime.now()
            session.current_step += 1

        return True

    def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Mark a session as complete and return summary statistics.

        Args:
            session_id: UUID of the session

        Returns:
            Dict with session statistics or empty dict if not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot complete: session {session_id} not found")
            return {}

        duration = datetime.now() - session.created_at
        stats = {
            "session_id": session_id,
            "model_id": session.model_id,
            "duration_seconds": duration.total_seconds(),
            "total_steps": session.total_steps,
            "steps_completed": session.current_step,
            "created_at": session.created_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

        logger.info(
            f"Completed session {session_id}: {session.current_step}/{session.total_steps} steps "
            f"in {duration.total_seconds():.2f}s"
        )
        return stats

    def cleanup_expired(self, timeout_seconds: Optional[int] = None) -> int:
        """
        Remove sessions that have exceeded the timeout.

        Args:
            timeout_seconds: Override timeout (uses default if None)

        Returns:
            Number of sessions cleaned up
        """
        timeout = timeout_seconds or self.timeout_seconds
        now = datetime.now()
        expired_sessions = []

        with self._lock:
            for session_id, session in list(self.sessions.items()):
                time_since_activity = now - session.last_activity
                if time_since_activity.total_seconds() > timeout:
                    expired_sessions.append((session_id, session))
                    del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            for session_id, session in expired_sessions:
                logger.debug(
                    f"Removed expired session {session_id} "
                    f"({(now - session.last_activity).total_seconds():.1f}s idle)"
                )

        return len(expired_sessions)

    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self.sessions)

    def get_sessions_info(self) -> Dict[str, Any]:
        """
        Get information about all active sessions.

        Returns:
            Dict with session summaries
        """
        with self._lock:
            sessions_info = {}
            for session_id, session in self.sessions.items():
                time_since_activity = datetime.now() - session.last_activity
                sessions_info[session_id] = {
                    "model_id": session.model_id,
                    "progress": f"{session.current_step}/{session.total_steps}",
                    "idle_seconds": time_since_activity.total_seconds(),
                    "created_at": session.created_at.isoformat(),
                }
            return sessions_info

    def _background_cleanup(self):
        """Background thread that periodically cleans up expired sessions."""
        logger.debug("Starting background cleanup thread")
        try:
            while True:
                time.sleep(self.cleanup_interval_seconds)
                self.cleanup_expired()
        except Exception as e:
            logger.error(f"Error in background cleanup thread: {e}", exc_info=True)

    def shutdown(self):
        """Gracefully shutdown the manager and cleanup thread."""
        logger.info("Shutting down SessionManager")
        # The daemon thread will auto-terminate with the process
        with self._lock:
            self.sessions.clear()
