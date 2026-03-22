# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI server for the N300 multi-modal web demo.

Provides REST endpoints for text/image/audio processing and WebSocket
for real-time tool status updates.

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/web_demo/server.py
"""

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# Imports from the agentic system
from models.demos.minimax_m2.agentic.loader import ModelBundle, cleanup_models, load_all_models, open_n300_device
from models.demos.minimax_m2.agentic.orchestrator import SYSTEM_PROMPT, _build_user_message, run_one_turn
from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS, dispatch_tool

# Configuration
UPLOAD_DIR = Path("/tmp/web_demo_uploads")
OUTPUT_DIR = Path("/tmp/web_demo_outputs")
STATIC_DIR = Path(__file__).parent / "static"
PORT = 7010

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# WebSocket manager for real-time updates
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections for broadcasting tool status."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        data = json.dumps(message)
        disconnected = set()
        # Copy the set to avoid "Set changed size during iteration"
        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.add(connection)
        for conn in disconnected:
            self.active_connections.discard(conn)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------


class AppState:
    """Global application state holding models and device."""

    def __init__(self):
        self.mesh_device = None
        self.models: Optional[ModelBundle] = None
        self.loading = False
        self.ready = False


state = AppState()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    text: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    want_audio_response: bool = False


class QueryResponse(BaseModel):
    text: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    tools_used: List[str] = []


class UploadResponse(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tenstorrent N300 Multi-Modal Demo",
    description="Web interface for the N300 agentic workflow",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Agentic query processing via LLM orchestrator
# ---------------------------------------------------------------------------


# Monkey-patch dispatch_tool to broadcast status via WebSocket
_original_dispatch_tool = dispatch_tool


def make_broadcast_dispatch_tool(broadcast_fn, tools_used_list):
    """Create a dispatch_tool wrapper that broadcasts status updates."""
    import asyncio

    tool_display_names = {
        "transcribe_audio": "Whisper STT",
        "text_to_speech": "Qwen3-TTS",
        "detect_objects": "OWL-ViT",
        "answer_from_context": "BERT QA",
        "generate_image": "Stable Diffusion",
        "detect_faces": "YUNet",
        "translate_text": "T5",
        "search_knowledge_base": "RAG (BGE)",
    }

    def broadcasting_dispatch_tool(name: str, args: Dict, models) -> Any:
        display_name = tool_display_names.get(name, name)
        tools_used_list.append(name)

        # Broadcast start (run in event loop)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    broadcast_fn({"tool": display_name, "status": "running", "args": str(args)[:100]})
                )
        except Exception:
            pass

        start_time = time.time()
        try:
            result = _original_dispatch_tool(name, args, models)
            duration = time.time() - start_time

            # Broadcast completion
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        broadcast_fn({"tool": display_name, "status": "done", "duration": round(duration, 2)})
                    )
            except Exception:
                pass

            return result
        except Exception as e:
            duration = time.time() - start_time
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        broadcast_fn(
                            {"tool": display_name, "status": "error", "error": str(e), "duration": round(duration, 2)}
                        )
                    )
            except Exception:
                pass
            raise

    return broadcasting_dispatch_tool


async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a query using the LLM orchestrator with tool calling."""
    import models.demos.minimax_m2.agentic.orchestrator as orchestrator_module
    import models.demos.minimax_m2.agentic.tools as tools_module

    tools_used = []
    response_audio_path = None
    response_image_path = None

    # Helper to broadcast
    async def broadcast(msg):
        await manager.broadcast(msg)

    # Build attachments list
    attachments = []
    if request.audio_path:
        attachments.append(request.audio_path)
    if request.image_path:
        attachments.append(request.image_path)

    # Build user text - if no text provided, give a default based on input type
    user_text = request.text.strip() if request.text else ""
    if not user_text:
        if request.audio_path and request.image_path:
            user_text = "What did I say in this audio, and what's in this image?"
        elif request.audio_path:
            user_text = "What did I say in this audio?"
        elif request.image_path:
            user_text = "What's in this image?"
        else:
            user_text = "Hello"

    # If user wants audio response, add it to the query
    if request.want_audio_response:
        user_text += " Please respond with audio."

    # Build the user message with attachment tags
    user_content = _build_user_message(user_text, attachments)

    # Create conversation with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Monkey-patch dispatch_tool in BOTH modules (tools and orchestrator)
    original_tools_dispatch = tools_module.dispatch_tool
    original_orchestrator_dispatch = orchestrator_module.dispatch_tool
    patched_dispatch = make_broadcast_dispatch_tool(broadcast, tools_used)
    tools_module.dispatch_tool = patched_dispatch
    orchestrator_module.dispatch_tool = patched_dispatch

    try:
        # Broadcast LLM start
        await broadcast({"tool": "LLM (Llama 8B)", "status": "running", "args": "Processing query..."})
        llm_start = time.time()

        # Run the agentic loop - LLM will call tools as needed
        response_text = run_one_turn(messages, state.models)

        llm_duration = time.time() - llm_start
        await broadcast({"tool": "LLM (Llama 8B)", "status": "done", "duration": round(llm_duration, 2)})

    finally:
        # Restore original dispatch_tool in both modules
        tools_module.dispatch_tool = original_tools_dispatch
        orchestrator_module.dispatch_tool = original_orchestrator_dispatch

    # Check if TTS was called and get the audio path
    if "text_to_speech" in tools_used:
        # Find the TTS output path from the response or use default
        response_audio_path = "/files/response.wav"
        # Check if file exists
        default_tts_path = Path("/tmp/response.wav")
        if default_tts_path.exists():
            # Copy to output dir with unique name
            output_filename = f"response_{uuid.uuid4().hex[:8]}.wav"
            import shutil

            shutil.copy(default_tts_path, OUTPUT_DIR / output_filename)
            response_audio_path = f"/files/{output_filename}"

    # Check if face detection was called and get the annotated image
    if "detect_faces" in tools_used:
        # Look for the annotated image in the response text
        import re

        match = re.search(r"Annotated image saved to: ([^\s]+)", response_text)
        if match:
            image_file = Path(match.group(1))
            if image_file.exists():
                response_image_path = f"/files/{image_file.name}"

    # Check if image generation was called
    if "generate_image" in tools_used:
        # Look for generated image path - default is /tmp/generated.png
        default_image_path = Path("/tmp/generated.png")
        if default_image_path.exists():
            # Copy to output dir with unique name
            output_filename = f"generated_{uuid.uuid4().hex[:8]}.png"
            shutil.copy(default_image_path, OUTPUT_DIR / output_filename)
            response_image_path = f"/files/{output_filename}"

    return QueryResponse(
        text=response_text,
        image_path=response_image_path,
        audio_path=response_audio_path,
        tools_used=tools_used,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup."""
    logger.info("Starting N300 web demo server...")
    logger.info(f"Static files: {STATIC_DIR}")
    logger.info(f"Upload dir: {UPLOAD_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Load models on startup
    logger.info("=" * 60)
    logger.info("Loading models on startup (this takes ~2-3 minutes)...")
    logger.info("=" * 60)

    try:
        state.loading = True
        state.mesh_device = open_n300_device(enable_fabric=True)

        state.models = load_all_models(
            state.mesh_device,
            load_llm=True,  # LLM orchestrator (Llama 3.1 8B)
            load_whisper=True,  # STT
            load_speecht5=True,  # TTS (fast, English only ~2.7s)
            load_qwen3_tts=False,  # TTS (slow, multi-language ~6min)
            load_owlvit=True,  # Object detection
            load_bert=True,  # QA
            load_sd=False,  # SD hangs when loaded with other models (works standalone)
            load_yunet=True,  # Face detection
            load_t5=True,  # Translation
            load_bge=True,  # BGE/TF-IDF embeddings for RAG (SBERT conflicts with LLM trace)
        )

        state.ready = True
        state.loading = False
        logger.info("=" * 60)
        logger.info("Models loaded! Server ready for inference.")
        logger.info(f"Open http://localhost:{PORT} in your browser")
        logger.info("=" * 60)

    except Exception as e:
        state.loading = False
        logger.exception(f"Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup models on shutdown."""
    if state.models is not None:
        logger.info("Cleaning up models...")
        cleanup_models(state.models)
    if state.mesh_device is not None:
        import ttnn

        logger.info("Closing mesh device...")
        ttnn.close_mesh_device(state.mesh_device)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Static files not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": state.ready,
        "models_loading": state.loading,
    }


@app.get("/status")
async def status():
    """Get detailed status of loaded models."""
    if not state.ready:
        return {"status": "not_loaded", "models": {}}

    models_status = {}
    rag_stats = None
    if state.models:
        models_status = {
            "llm": state.models.llm is not None,
            "whisper": state.models.whisper is not None,
            "speecht5": state.models.speecht5 is not None,
            "qwen3_tts": state.models.qwen3_tts is not None,
            "owlvit": state.models.owlvit is not None,
            "bert": state.models.bert is not None,
            "t5": state.models.t5 is not None,
            "yunet": state.models.yunet is not None,
            "sd": state.models.sd is not None,
            "bge": state.models.bge is not None,  # TF-IDF embeddings (SBERT conflicts with LLM)
            "rag": state.models.rag is not None,
        }
        # Include RAG stats if available
        if state.models.rag is not None:
            rag_stats = state.models.rag.stats()
    return {"status": "ready", "models": models_status, "rag": rag_stats}


@app.get("/tools")
async def get_tools():
    """Return the list of available tool schemas."""
    return TOOL_SCHEMAS


@app.post("/load-models")
async def load_models_endpoint():
    """Load models on demand (takes ~2-3 minutes)."""
    if state.ready:
        return {"status": "already_loaded"}
    if state.loading:
        return {"status": "loading_in_progress"}

    state.loading = True

    async def broadcast(msg):
        await manager.broadcast(msg)

    try:
        await broadcast({"tool": "System", "status": "running", "args": "Opening N300 mesh device..."})
        state.mesh_device = open_n300_device(enable_fabric=True)

        await broadcast({"tool": "System", "status": "running", "args": "Loading models (this takes ~2-3 min)..."})
        state.models = load_all_models(
            state.mesh_device,
            load_llm=False,  # Skip LLM for faster startup
            load_whisper=True,
            load_speecht5=True,  # Fast TTS
            load_qwen3_tts=False,
            load_owlvit=True,
            load_bert=True,
            load_sd=False,  # Skip SD for faster startup
            load_yunet=False,
            load_t5=False,
            load_bge=True,  # BGE/TF-IDF for RAG (SBERT conflicts with LLM trace)
        )

        state.ready = True
        state.loading = False
        await broadcast({"tool": "System", "status": "done", "args": "Models loaded successfully!"})
        return {"status": "loaded"}

    except Exception as e:
        state.loading = False
        await broadcast({"tool": "System", "status": "error", "error": str(e)})
        logger.exception("Failed to load models")
        return {"status": "error", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a multi-modal query."""
    if not state.ready:
        return QueryResponse(
            text="Models not loaded. Click 'Load Models' first.",
            tools_used=[],
        )

    # Validate input - require at least some content
    has_text = request.text and request.text.strip()
    has_image = request.image_path is not None
    has_audio = request.audio_path is not None

    if not has_text and not has_image and not has_audio:
        return QueryResponse(
            text="Please provide a question, image, or audio input.",
            tools_used=[],
        )

    try:
        return await process_query(request)
    except Exception as e:
        logger.exception("Query failed")
        return QueryResponse(text=f"Error: {str(e)}", tools_used=[])


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload an image or audio file."""
    # Generate unique filename
    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    filepath = UPLOAD_DIR / filename

    # Save file
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded: {filepath}")
    return UploadResponse(path=str(filepath))


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve generated output files."""
    # Check both output and upload directories
    for directory in [OUTPUT_DIR, UPLOAD_DIR]:
        filepath = directory / filename
        if filepath.exists():
            media_type = None
            if filename.endswith(".wav"):
                media_type = "audio/wav"
            elif filename.endswith(".png"):
                media_type = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                media_type = "image/jpeg"
            return FileResponse(str(filepath), media_type=media_type)
    return {"error": "File not found"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time tool status updates."""
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_json(
            {
                "tool": "System",
                "status": "connected",
                "models_loaded": state.ready,
            }
        )

        # Keep connection alive and handle any incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Handle ping/pong or other client messages
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# RAG endpoints
# ---------------------------------------------------------------------------


class RAGTextRequest(BaseModel):
    text: str
    source: str = "user_input"


class RAGResponse(BaseModel):
    status: str
    chunks_added: int = 0
    message: str = ""


@app.post("/rag/upload", response_model=RAGResponse)
async def rag_upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG knowledge base."""
    if not state.ready or state.models is None or state.models.rag is None:
        return RAGResponse(status="error", message="RAG not loaded. Enable BGE in server config.")

    # Save file temporarily
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".md", ".py", ".json", ".yaml", ".yml"]:
        return RAGResponse(status="error", message=f"Unsupported file type: {ext}")

    filename = f"rag_{uuid.uuid4().hex[:12]}{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Add to RAG
    try:
        chunks_added = state.models.rag.add_file(str(filepath))
        await manager.broadcast(
            {"tool": "RAG", "status": "done", "args": f"Added {chunks_added} chunks from {file.filename}"}
        )
        return RAGResponse(status="ok", chunks_added=chunks_added, message=f"Added {file.filename}")
    except Exception as e:
        return RAGResponse(status="error", message=str(e))


@app.post("/rag/add-text", response_model=RAGResponse)
async def rag_add_text(request: RAGTextRequest):
    """Add text directly to the RAG knowledge base."""
    if not state.ready or state.models is None or state.models.rag is None:
        return RAGResponse(status="error", message="RAG not loaded. Enable BGE in server config.")

    try:
        chunks_added = state.models.rag.add_document(request.text, source=request.source)
        await manager.broadcast({"tool": "RAG", "status": "done", "args": f"Added {chunks_added} chunks"})
        return RAGResponse(status="ok", chunks_added=chunks_added, message=f"Added {chunks_added} chunks")
    except Exception as e:
        return RAGResponse(status="error", message=str(e))


@app.get("/rag/stats")
async def rag_stats():
    """Get RAG knowledge base statistics."""
    if not state.ready or state.models is None or state.models.rag is None:
        return {"status": "not_loaded", "total_chunks": 0, "sources": {}}

    return state.models.rag.stats()


@app.post("/rag/clear")
async def rag_clear():
    """Clear the RAG knowledge base."""
    if not state.ready or state.models is None or state.models.rag is None:
        return {"status": "error", "message": "RAG not loaded"}

    state.models.rag.clear()
    return {"status": "ok", "message": "Knowledge base cleared"}


@app.post("/rag/search")
async def rag_search(query: str, top_k: int = 3):
    """Search the RAG knowledge base directly (for testing)."""
    if not state.ready or state.models is None or state.models.rag is None:
        return {"status": "error", "message": "RAG not loaded"}

    results = state.models.rag.search(query, top_k=top_k)
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}...")
    logger.info(f"Open http://localhost:{PORT} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
