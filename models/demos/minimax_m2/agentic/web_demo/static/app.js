/**
 * Tenstorrent N300 Multi-Modal Demo - Frontend JavaScript
 */

// ============================================================================
// State
// ============================================================================

const state = {
    ws: null,
    modelsLoaded: false,
    recording: false,
    mediaRecorder: null,
    audioChunks: [],
    uploadedImagePath: null,
    uploadedAudioPath: null,
    ragDocuments: 0,
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Header
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),

    // Input
    textInput: document.getElementById('text-input'),
    imageInput: document.getElementById('image-input'),
    imageDropZone: document.getElementById('image-drop-zone'),
    imagePreview: document.getElementById('image-preview'),
    clearImageBtn: document.getElementById('clear-image-btn'),
    audioInput: document.getElementById('audio-input'),
    audioDropZone: document.getElementById('audio-drop-zone'),
    audioPreview: document.getElementById('audio-preview'),
    audioStatus: document.getElementById('audio-status'),
    clearAudioBtn: document.getElementById('clear-audio-btn'),
    recordBtn: document.getElementById('record-btn'),
    wantAudioResponse: document.getElementById('want-audio-response'),
    submitBtn: document.getElementById('submit-btn'),

    // Output
    textOutput: document.getElementById('text-output'),
    imageOutputGroup: document.getElementById('image-output-group'),
    imageOutput: document.getElementById('image-output'),
    audioOutputGroup: document.getElementById('audio-output-group'),
    audioOutput: document.getElementById('audio-output'),
    toolsUsed: document.getElementById('tools-used'),

    // Console
    console: document.getElementById('console'),
    clearConsoleBtn: document.getElementById('clear-console-btn'),

    // RAG
    ragToggle: document.getElementById('rag-toggle'),
    ragContent: document.getElementById('rag-content'),
    ragInput: document.getElementById('rag-input'),
    ragDropZone: document.getElementById('rag-drop-zone'),
    ragFiles: document.getElementById('rag-files'),
    ragStatus: document.getElementById('rag-status'),
    clearRagBtn: document.getElementById('clear-rag-btn'),
};

// ============================================================================
// WebSocket Connection
// ============================================================================

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        updateStatus('connected', 'Connected');
        addConsoleLog('System', 'WebSocket connected', 'system');
    };

    state.ws.onclose = () => {
        updateStatus('offline', 'Disconnected');
        addConsoleLog('System', 'WebSocket disconnected. Reconnecting...', 'error');
        // Reconnect after 2 seconds
        setTimeout(initWebSocket, 2000);
    };

    state.ws.onerror = (error) => {
        addConsoleLog('System', `WebSocket error: ${error.message || 'Unknown'}`, 'error');
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    // Send periodic pings to keep connection alive
    setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send('ping');
        }
    }, 25000);
}

function handleWebSocketMessage(data) {
    // Handle different message types
    if (data.type === 'pong' || data.type === 'heartbeat') {
        return; // Ignore heartbeat/pong messages
    }

    // Handle tool status updates
    if (data.tool) {
        const status = data.status || 'info';
        let message = '';

        if (status === 'running') {
            message = data.args ? `Running: ${data.args}` : 'Running...';
        } else if (status === 'done') {
            message = data.duration ? `Done (${data.duration}s)` : 'Done';
        } else if (status === 'error') {
            message = data.error ? `Error: ${data.error}` : 'Error occurred';
        } else if (status === 'connected') {
            state.modelsLoaded = data.models_loaded || false;
            updateModelStatus();
            return;
        } else {
            message = data.args || data.message || status;
        }

        addConsoleLog(data.tool, message, status);

        // Update model status if system message about loading
        if (data.tool === 'System' && data.args && data.args.includes('loaded')) {
            state.modelsLoaded = true;
            updateModelStatus();
        }
    }
}

function updateStatus(state, text) {
    elements.statusIndicator.className = `status-dot ${state}`;
    elements.statusText.textContent = text;
}

function updateModelStatus() {
    if (state.modelsLoaded) {
        updateStatus('ready', 'Ready');
        elements.submitBtn.disabled = false;
    } else {
        updateStatus('loading', 'Loading models...');
        elements.submitBtn.disabled = true;
    }
}

// ============================================================================
// Console Logging
// ============================================================================

function addConsoleLog(tool, message, status = 'info') {
    const entry = document.createElement('div');
    entry.className = `console-entry ${status}`;

    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });

    entry.innerHTML = `
        <span class="timestamp">[${timestamp}]</span>
        <span class="tool">${tool}</span>
        <span class="message">${escapeHtml(message)}</span>
    `;

    elements.console.appendChild(entry);
    elements.console.scrollTop = elements.console.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// File Upload
// ============================================================================

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        return data.path;
    } catch (error) {
        addConsoleLog('System', `Upload error: ${error.message}`, 'error');
        throw error;
    }
}

function setupImageUpload() {
    // Click to upload
    elements.imageDropZone.addEventListener('click', () => {
        elements.imageInput.click();
    });

    // File input change
    elements.imageInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await handleImageFile(file);
        }
    });

    // Drag and drop
    elements.imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.imageDropZone.classList.add('dragover');
    });

    elements.imageDropZone.addEventListener('dragleave', () => {
        elements.imageDropZone.classList.remove('dragover');
    });

    elements.imageDropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        elements.imageDropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            await handleImageFile(file);
        }
    });

    // Clear button
    elements.clearImageBtn.addEventListener('click', () => {
        clearImage();
    });
}

async function handleImageFile(file) {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.imagePreview.src = e.target.result;
        elements.imagePreview.classList.remove('hidden');
        elements.clearImageBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Upload
    try {
        addConsoleLog('System', `Uploading image: ${file.name}`, 'running');
        state.uploadedImagePath = await uploadFile(file);
        addConsoleLog('System', 'Image uploaded', 'done');
    } catch (error) {
        clearImage();
    }
}

function clearImage() {
    state.uploadedImagePath = null;
    elements.imagePreview.src = '';
    elements.imagePreview.classList.add('hidden');
    elements.clearImageBtn.classList.add('hidden');
    elements.imageInput.value = '';
}

function setupAudioUpload() {
    // Click to upload
    elements.audioDropZone.addEventListener('click', () => {
        elements.audioInput.click();
    });

    // File input change
    elements.audioInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await handleAudioFile(file);
        }
    });

    // Drag and drop
    elements.audioDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.audioDropZone.classList.add('dragover');
    });

    elements.audioDropZone.addEventListener('dragleave', () => {
        elements.audioDropZone.classList.remove('dragover');
    });

    elements.audioDropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        elements.audioDropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            await handleAudioFile(file);
        }
    });

    // Clear button
    elements.clearAudioBtn.addEventListener('click', () => {
        clearAudio();
    });
}

async function handleAudioFile(file) {
    // Show preview
    const url = URL.createObjectURL(file);
    elements.audioPreview.src = url;
    elements.audioPreview.classList.remove('hidden');
    elements.clearAudioBtn.classList.remove('hidden');

    // Upload
    try {
        addConsoleLog('System', `Uploading audio: ${file.name}`, 'running');
        state.uploadedAudioPath = await uploadFile(file);
        addConsoleLog('System', 'Audio uploaded', 'done');
    } catch (error) {
        clearAudio();
    }
}

function clearAudio() {
    state.uploadedAudioPath = null;
    elements.audioPreview.src = '';
    elements.audioPreview.classList.add('hidden');
    elements.audioStatus.classList.add('hidden');
    elements.clearAudioBtn.classList.add('hidden');
    elements.audioInput.value = '';
}

// ============================================================================
// Microphone Recording
// ============================================================================

function setupRecording() {
    elements.recordBtn.addEventListener('click', async () => {
        if (state.recording) {
            stopRecording();
        } else {
            await startRecording();
        }
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        state.audioChunks = [];
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };

        state.mediaRecorder.onstop = async () => {
            // Convert to WAV and upload
            const webmBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            const wavBlob = await convertToWav(webmBlob);
            const file = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
            await handleAudioFile(file);

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        state.mediaRecorder.start();
        state.recording = true;
        elements.recordBtn.textContent = 'Stop';
        elements.recordBtn.classList.add('recording');
        elements.audioStatus.textContent = 'Recording...';
        elements.audioStatus.classList.remove('hidden');
        elements.audioStatus.classList.add('recording');

        addConsoleLog('System', 'Recording started', 'running');
    } catch (error) {
        addConsoleLog('System', `Microphone error: ${error.message}`, 'error');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.recording) {
        state.mediaRecorder.stop();
        state.recording = false;
        elements.recordBtn.textContent = 'Record';
        elements.recordBtn.classList.remove('recording');
        elements.audioStatus.textContent = 'Processing...';
        elements.audioStatus.classList.remove('recording');

        addConsoleLog('System', 'Recording stopped', 'done');
    }
}

async function convertToWav(webmBlob) {
    // For simplicity, we'll upload webm and let the server handle conversion
    // In production, you'd use an AudioContext to properly convert to WAV
    // For now, just return a renamed blob (the server's ffmpeg can handle webm)
    return new Blob([webmBlob], { type: 'audio/wav' });
}

// ============================================================================
// Query Submission
// ============================================================================

function setupSubmit() {
    elements.submitBtn.addEventListener('click', submitQuery);

    // Also submit on Ctrl+Enter in text area
    elements.textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            submitQuery();
        }
    });
}

async function submitQuery() {
    const text = elements.textInput.value.trim();
    const hasInput = text || state.uploadedImagePath || state.uploadedAudioPath;

    if (!hasInput) {
        addConsoleLog('System', 'Please provide some input (text, image, or audio)', 'error');
        return;
    }

    if (!state.modelsLoaded) {
        addConsoleLog('System', 'Models not loaded. Click "Load Models" first.', 'error');
        return;
    }

    // Disable submit button during processing
    elements.submitBtn.disabled = true;
    elements.submitBtn.textContent = 'Processing...';

    // Clear previous output
    elements.textOutput.innerHTML = '<p class="placeholder">Processing...</p>';

    // Reset image output
    elements.imageOutput.classList.add('hidden');
    elements.imageOutput.src = '';
    const imagePlaceholder = document.getElementById('image-placeholder');
    if (imagePlaceholder) imagePlaceholder.classList.remove('hidden');

    // Reset audio output
    elements.audioOutput.classList.add('hidden');
    elements.audioOutput.src = '';
    const audioPlaceholder = document.getElementById('audio-placeholder');
    if (audioPlaceholder) audioPlaceholder.classList.remove('hidden');

    elements.toolsUsed.innerHTML = '<span class="placeholder">-</span>';

    try {
        addConsoleLog('System', 'Submitting query...', 'running');

        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                image_path: state.uploadedImagePath,
                audio_path: state.uploadedAudioPath,
                want_audio_response: elements.wantAudioResponse.checked,
            }),
        });

        if (!response.ok) {
            throw new Error(`Query failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResponse(data);

        addConsoleLog('System', 'Query completed', 'done');
    } catch (error) {
        addConsoleLog('System', `Query error: ${error.message}`, 'error');
        elements.textOutput.innerHTML = `<p style="color: var(--accent-error);">Error: ${escapeHtml(error.message)}</p>`;
    } finally {
        elements.submitBtn.disabled = false;
        elements.submitBtn.textContent = 'Submit';
    }
}

function displayResponse(data) {
    // Text response
    if (data.text) {
        elements.textOutput.innerHTML = '';
        const pre = document.createElement('pre');
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.margin = '0';
        pre.textContent = data.text;
        elements.textOutput.appendChild(pre);
    } else {
        elements.textOutput.innerHTML = '<p class="placeholder">No text response</p>';
    }

    // Image output
    const imagePlaceholder = document.getElementById('image-placeholder');
    if (data.image_path) {
        elements.imageOutput.src = data.image_path;
        elements.imageOutput.classList.remove('hidden');
        if (imagePlaceholder) imagePlaceholder.classList.add('hidden');
    } else {
        elements.imageOutput.classList.add('hidden');
        if (imagePlaceholder) imagePlaceholder.classList.remove('hidden');
    }

    // Audio output
    const audioPlaceholder = document.getElementById('audio-placeholder');
    if (data.audio_path) {
        elements.audioOutput.src = data.audio_path;
        elements.audioOutput.classList.remove('hidden');
        if (audioPlaceholder) audioPlaceholder.classList.add('hidden');
    } else {
        elements.audioOutput.classList.add('hidden');
        if (audioPlaceholder) audioPlaceholder.classList.remove('hidden');
    }

    // Tools used
    if (data.tools_used && data.tools_used.length > 0) {
        elements.toolsUsed.innerHTML = data.tools_used
            .map(tool => `<span class="tool-tag">${escapeHtml(tool)}</span>`)
            .join('');
    } else {
        elements.toolsUsed.innerHTML = '<span class="placeholder">-</span>';
    }
}

// ============================================================================
// Clear Console
// ============================================================================

function setupClearConsole() {
    elements.clearConsoleBtn.addEventListener('click', () => {
        elements.console.innerHTML = '';
        addConsoleLog('System', 'Console cleared', 'system');
    });
}

// ============================================================================
// RAG (Knowledge Base)
// ============================================================================

function setupRAG() {
    // Toggle collapse/expand
    elements.ragToggle.addEventListener('click', () => {
        elements.ragContent.classList.toggle('hidden');
        elements.ragToggle.classList.toggle('expanded');
    });

    // Click to upload
    elements.ragDropZone.addEventListener('click', () => {
        elements.ragInput.click();
    });

    // File input change
    elements.ragInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        for (const file of files) {
            await handleRagFile(file);
        }
        elements.ragInput.value = '';
    });

    // Drag and drop
    elements.ragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.ragDropZone.classList.add('dragover');
    });

    elements.ragDropZone.addEventListener('dragleave', () => {
        elements.ragDropZone.classList.remove('dragover');
    });

    elements.ragDropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        elements.ragDropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        for (const file of files) {
            await handleRagFile(file);
        }
    });

    // Clear button
    elements.clearRagBtn.addEventListener('click', clearRAG);
}

async function handleRagFile(file) {
    const allowedExtensions = ['.txt', '.md', '.py', '.json', '.yaml', '.yml'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(ext)) {
        addConsoleLog('RAG', `Unsupported file type: ${ext}`, 'error');
        return;
    }

    try {
        addConsoleLog('RAG', `Adding document: ${file.name}`, 'running');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/rag/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        addConsoleLog('RAG', `Added ${data.chunks_added} chunks from ${file.name}`, 'done');

        // Update RAG stats display
        await updateRagStats();

        // Add file to display
        addRagFileEntry(file.name, data.chunks_added);

    } catch (error) {
        addConsoleLog('RAG', `Error: ${error.message}`, 'error');
    }
}

function addRagFileEntry(filename, chunks) {
    // Remove placeholder if present
    const placeholder = elements.ragFiles.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const entry = document.createElement('div');
    entry.className = 'rag-file';
    entry.innerHTML = `
        <span class="filename">${escapeHtml(filename)}</span>
        <span class="chunks">${chunks} chunks</span>
    `;
    elements.ragFiles.appendChild(entry);
}

async function updateRagStats() {
    try {
        const response = await fetch('/rag/stats');
        if (response.ok) {
            const data = await response.json();
            state.ragDocuments = data.total_chunks || 0;
            elements.ragStatus.textContent = `${state.ragDocuments} chunks`;
        }
    } catch (error) {
        console.error('Failed to fetch RAG stats:', error);
    }
}

async function clearRAG() {
    try {
        addConsoleLog('RAG', 'Clearing knowledge base...', 'running');

        const response = await fetch('/rag/clear', { method: 'POST' });

        if (!response.ok) {
            throw new Error(`Clear failed: ${response.statusText}`);
        }

        state.ragDocuments = 0;
        elements.ragStatus.textContent = '0 chunks';
        elements.ragFiles.innerHTML = '<span class="placeholder">No documents uploaded</span>';

        addConsoleLog('RAG', 'Knowledge base cleared', 'done');
    } catch (error) {
        addConsoleLog('RAG', `Error: ${error.message}`, 'error');
    }
}

// ============================================================================
// Initialize
// ============================================================================

function init() {
    // Setup all event handlers
    setupImageUpload();
    setupAudioUpload();
    setupRecording();
    setupSubmit();
    setupClearConsole();
    setupRAG();

    // Connect WebSocket
    initWebSocket();

    // Check health and poll until models are loaded
    checkHealth();

    addConsoleLog('System', 'Frontend initialized', 'system');
    addConsoleLog('System', 'Waiting for models to load...', 'running');
}

async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.models_loaded) {
            state.modelsLoaded = true;
            updateModelStatus();
            addConsoleLog('System', 'Models ready! You can start querying.', 'done');
        } else if (data.models_loading) {
            updateStatus('loading', 'Loading models...');
            // Poll again in 2 seconds
            setTimeout(checkHealth, 2000);
        } else {
            // Server starting up, poll again
            setTimeout(checkHealth, 2000);
        }
    } catch (error) {
        // Server not ready yet, poll again
        updateStatus('offline', 'Waiting for server...');
        setTimeout(checkHealth, 2000);
    }
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
