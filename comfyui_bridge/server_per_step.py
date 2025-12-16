"""
Per-Step API Server Operations for ComfyUI Bridge

This module provides operation routing, session validation, and error handling
for the per-step denoising API. Designed to be imported and integrated into
existing server code.

Usage:
    from comfyui_bridge.server_per_step import register_per_step_operations

    # In your server initialization:
    register_per_step_operations(app)
"""

import logging
from typing import Dict, Any, Callable, Optional
from functools import wraps
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class PerStepOperationRegistry:
    """Registry for per-step API operations with middleware support."""

    def __init__(self):
        self.operations: Dict[str, Callable] = {}
        self.session_store: Dict[str, Any] = {}
        self.middleware_chain = []
        self.handlers: Optional[Any] = None  # Will be set to PerStepHandlers instance
        self.model_registry: Dict[str, Any] = {}  # Models available for inference
        self.scheduler_registry: Dict[str, Any] = {}  # Schedulers available

    def register_operation(self, operation_name: str, handler: Callable):
        """Register an operation handler."""
        self.operations[operation_name] = handler
        logger.info(f"Registered per-step operation: {operation_name}")

    def add_middleware(self, middleware: Callable):
        """Add middleware to the processing chain."""
        self.middleware_chain.append(middleware)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by ID."""
        return self.session_store.get(session_id)

    def create_session(self, session_id: str, session_data: Dict[str, Any]):
        """Create a new session."""
        self.session_store[session_id] = session_data
        logger.info(f"Created session: {session_id}")

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"Deleted session: {session_id}")

    def set_handlers(self, handlers: Any):
        """Set the PerStepHandlers instance."""
        self.handlers = handlers
        logger.info(f"Registered PerStepHandlers: {type(handlers).__name__}")

    def set_model_registry(self, models: Dict[str, Any]):
        """Set the model registry."""
        self.model_registry = models
        logger.info(f"Registered {len(models)} models")

    def set_scheduler_registry(self, schedulers: Dict[str, Any]):
        """Set the scheduler registry."""
        self.scheduler_registry = schedulers
        logger.info(f"Registered {len(schedulers)} schedulers")

    def dispatch(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an operation through the middleware chain."""
        if operation not in self.operations:
            raise ValueError(f"Unknown operation: {operation}")

        # Check if this is a handlers-based operation and delegate
        handlers_ops = ["session_create", "denoise_step_single", "session_complete", "session_status", "session_cleanup"]
        if operation in handlers_ops and self.handlers:
            logger.debug(f"Delegating {operation} to PerStepHandlers")
            return self._dispatch_to_handlers(operation, params)

        # Execute middleware chain for legacy operations
        context = {
            "operation": operation,
            "params": params,
            "registry": self
        }

        for middleware in self.middleware_chain:
            try:
                middleware(context)
            except Exception as e:
                raise RuntimeError(f"Middleware failed: {str(e)}") from e

        # Execute the operation handler
        handler = self.operations[operation]
        return handler(params, context)

    def _dispatch_to_handlers(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch operation to PerStepHandlers instance."""
        handler_methods = {
            "session_create": self.handlers.handle_session_create,
            "denoise_step_single": self.handlers.handle_denoise_step_single,
            "session_complete": self.handlers.handle_session_complete,
            "session_status": self.handlers.handle_session_status,
            "session_cleanup": self.handlers.handle_session_cleanup,
        }

        if operation not in handler_methods:
            raise ValueError(f"Handler not found for operation: {operation}")

        handler = handler_methods[operation]
        return handler(params)


# Global registry instance
_registry = PerStepOperationRegistry()


# ============================================================================
# Middleware Functions
# ============================================================================

def session_validation_middleware(context: Dict[str, Any]):
    """
    Validate session existence for operations that require it.

    Operations requiring session validation:
    - denoise_step_single
    - session_complete
    """
    operation = context["operation"]
    params = context["params"]
    registry = context["registry"]

    # Operations that require session validation
    session_required_ops = ["denoise_step_single", "session_complete"]

    if operation in session_required_ops:
        session_id = params.get("session_id")

        if not session_id:
            raise ValueError(f"Operation '{operation}' requires 'session_id' parameter")

        session_data = registry.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")

        # Add session data to context for handler use
        context["session_data"] = session_data
        logger.debug(f"Session validated: {session_id}")


def request_logging_middleware(context: Dict[str, Any]):
    """Log incoming requests."""
    operation = context["operation"]
    params = context["params"]
    logger.info(f"Per-step API request: {operation}")
    logger.debug(f"Request parameters: {params}")


def error_handling_middleware(context: Dict[str, Any]):
    """
    Validate common parameters and handle errors.
    Note: This runs before the operation handler.
    """
    params = context["params"]

    # Validate common parameter types
    if "session_id" in params and not isinstance(params["session_id"], str):
        raise TypeError("'session_id' must be a string")

    if "step" in params and not isinstance(params["step"], int):
        raise TypeError("'step' must be an integer")


# ============================================================================
# Operation Handlers
# ============================================================================

def handle_session_create(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new denoising session.

    Required params:
        - session_id: Unique session identifier
        - prompt: Text prompt for generation
        - num_inference_steps: Total number of denoising steps

    Optional params:
        - negative_prompt: Negative prompt text
        - guidance_scale: CFG guidance scale (default: 7.5)
        - seed: Random seed for generation
        - width: Image width (default: 1024)
        - height: Image height (default: 1024)
    """
    registry = context["registry"]

    # Validate required parameters
    required = ["session_id", "prompt", "num_inference_steps"]
    for param in required:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")

    session_id = params["session_id"]

    # Check for duplicate session
    if registry.get_session(session_id):
        raise ValueError(f"Session already exists: {session_id}")

    # Create session data structure
    session_data = {
        "session_id": session_id,
        "prompt": params["prompt"],
        "negative_prompt": params.get("negative_prompt", ""),
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params.get("guidance_scale", 7.5),
        "seed": params.get("seed"),
        "width": params.get("width", 1024),
        "height": params.get("height", 1024),
        "current_step": 0,
        "state": "initialized",
        "latents": None,  # Will be initialized by backend
        "pipeline_state": None  # Backend-specific state
    }

    registry.create_session(session_id, session_data)

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session created successfully",
        "session_info": {
            "num_inference_steps": session_data["num_inference_steps"],
            "guidance_scale": session_data["guidance_scale"],
            "dimensions": f"{session_data['width']}x{session_data['height']}"
        }
    }


def handle_denoise_step_single(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single denoising step for a session.

    Required params:
        - session_id: Session identifier
        - step: Step number to execute (0-indexed)

    Optional params:
        - return_latent: Return latent tensor data (default: False)
        - return_preview: Return decoded preview image (default: False)
    """
    registry = context["registry"]
    session_data = context["session_data"]  # Provided by middleware

    step = params.get("step")
    if step is None:
        raise ValueError("Missing required parameter: step")

    num_steps = session_data["num_inference_steps"]
    if not (0 <= step < num_steps):
        raise ValueError(f"Step {step} out of range [0, {num_steps})")

    # Check step sequencing
    current_step = session_data["current_step"]
    if step != current_step:
        logger.warning(
            f"Step mismatch: requested {step}, expected {current_step}. "
            f"Allowing out-of-order execution."
        )

    # This is where backend integration would occur
    # For now, return a structured response that indicates backend call needed

    # Simulate step execution
    session_data["current_step"] = step + 1
    session_data["state"] = "in_progress" if step < num_steps - 1 else "completed"

    response = {
        "success": True,
        "session_id": session_data["session_id"],
        "step": step,
        "total_steps": num_steps,
        "is_final_step": step == num_steps - 1,
        "state": session_data["state"]
    }

    # Optional return values
    if params.get("return_latent", False):
        response["latent"] = {
            "shape": [1, 4, session_data["height"] // 8, session_data["width"] // 8],
            "dtype": "float32",
            "data": None  # Backend would provide actual data
        }

    if params.get("return_preview", False):
        response["preview"] = {
            "format": "base64_jpeg",
            "data": None  # Backend would provide actual image
        }

    return response


def handle_session_complete(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete a session and return final results.

    Required params:
        - session_id: Session identifier

    Optional params:
        - return_image: Return final decoded image (default: True)
        - cleanup: Delete session after completion (default: True)
    """
    registry = context["registry"]
    session_data = context["session_data"]  # Provided by middleware

    session_id = session_data["session_id"]

    # Check if all steps completed
    current_step = session_data["current_step"]
    num_steps = session_data["num_inference_steps"]

    if current_step < num_steps:
        logger.warning(
            f"Session {session_id} completing with {current_step}/{num_steps} steps. "
            f"Not all steps executed."
        )

    # Decode final image (backend integration point)
    response = {
        "success": True,
        "session_id": session_id,
        "steps_completed": current_step,
        "total_steps": num_steps
    }

    if params.get("return_image", True):
        response["image"] = {
            "format": "base64_png",
            "width": session_data["width"],
            "height": session_data["height"],
            "data": None  # Backend would provide actual image
        }

    # Cleanup session if requested
    if params.get("cleanup", True):
        registry.delete_session(session_id)
        response["session_cleaned_up"] = True

    return response


# ============================================================================
# Response Formatting
# ============================================================================

def format_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format a successful operation response."""
    return {
        "status": "success",
        "data": data,
        "error": None
    }


def format_error_response(error: Exception, operation: str) -> Dict[str, Any]:
    """Format an error response with details."""
    error_type = type(error).__name__
    error_message = str(error)

    response = {
        "status": "error",
        "data": None,
        "error": {
            "type": error_type,
            "message": error_message,
            "operation": operation
        }
    }

    # Include traceback for unexpected errors (not validation errors)
    if not isinstance(error, (ValueError, TypeError)):
        response["error"]["traceback"] = traceback.format_exc()

    logger.error(f"Operation '{operation}' failed: {error_type}: {error_message}")

    return response


# ============================================================================
# Public API
# ============================================================================

def register_per_step_operations(server_app=None, handlers=None, models=None, schedulers=None):
    """
    Register all per-step operations with the registry.

    Args:
        server_app: Optional server application instance for framework-specific integration
                   (e.g., Flask app, FastAPI app, etc.)
        handlers: Optional PerStepHandlers instance for per-step operations
        models: Optional dict of available models {model_id: model_obj}
        schedulers: Optional dict of available schedulers {scheduler_name: scheduler_obj}

    Returns:
        The configured registry instance

    Usage:
        from comfyui_bridge.handlers_per_step import PerStepHandlers
        handlers = PerStepHandlers(models_dict, schedulers_dict)
        registry = register_per_step_operations(
            server_app=app,
            handlers=handlers,
            models=models_dict,
            schedulers=schedulers_dict
        )
    """
    global _registry

    # Register middleware (order matters)
    _registry.add_middleware(request_logging_middleware)
    _registry.add_middleware(error_handling_middleware)
    _registry.add_middleware(session_validation_middleware)

    # Register operation handlers
    _registry.register_operation("session_create", handle_session_create)
    _registry.register_operation("denoise_step_single", handle_denoise_step_single)
    _registry.register_operation("session_complete", handle_session_complete)
    _registry.register_operation("session_status", lambda p, c: {"error": "Not implemented"})
    _registry.register_operation("session_cleanup", lambda p, c: {"error": "Not implemented"})

    # Set up handlers-based infrastructure if provided
    if handlers:
        _registry.set_handlers(handlers)
        logger.info("PerStepHandlers configured for per-step operations")

    if models:
        _registry.set_model_registry(models)

    if schedulers:
        _registry.set_scheduler_registry(schedulers)

    logger.info("Per-step operations registered successfully")

    # If server app provided, can add framework-specific integration here
    if server_app:
        logger.info(f"Server app integration: {type(server_app).__name__}")
        # Example: app.add_route("/per_step", handle_per_step_request)

    return _registry


def handle_per_step_request(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for handling per-step API requests.

    Args:
        operation: Operation name (e.g., "session_create", "denoise_step_single")
        params: Operation parameters as dictionary

    Returns:
        Formatted response dictionary with status, data, and error fields
    """
    try:
        result = _registry.dispatch(operation, params)
        return format_success_response(result)
    except Exception as e:
        return format_error_response(e, operation)


def get_registry() -> PerStepOperationRegistry:
    """Get the global registry instance."""
    return _registry


def reset_registry():
    """Reset the registry (useful for testing)."""
    global _registry
    _registry = PerStepOperationRegistry()
    logger.info("Registry reset")


# ============================================================================
# Utility Functions
# ============================================================================

def list_operations() -> list[str]:
    """List all registered operations."""
    return list(_registry.operations.keys())


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_registry.session_store.keys())


def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get session information (safe subset).

    Returns public session info without internal state.
    """
    session = _registry.get_session(session_id)
    if not session:
        return None

    return {
        "session_id": session["session_id"],
        "prompt": session["prompt"],
        "num_inference_steps": session["num_inference_steps"],
        "current_step": session["current_step"],
        "state": session["state"],
        "dimensions": f"{session['width']}x{session['height']}"
    }
